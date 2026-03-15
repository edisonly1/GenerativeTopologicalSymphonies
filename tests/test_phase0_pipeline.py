"""Tests for the initial preprocessing and tokenization pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from inference import render_piece_to_midi_bytes
from preprocessing import (
    ParsedPiece,
    annotate_quantized_piece_harmony,
    detect_asap_dataset,
    detect_dataset_kind,
    detect_maestro_official_splits,
    generate_toy_dataset,
    load_quantized_piece_json,
    parse_midi_bytes,
    prepare_dataset,
    quantize_piece,
    segment_phrases,
)
from preprocessing.schema import NoteEvent, QuantizedEvent, QuantizedPiece
from training.data import (
    AutoregressiveTokenDataset,
    collate_autoregressive_batch,
    create_autoregressive_dataloader,
)
from tokenization import (
    encode_piece_to_blocks,
    load_processed_dataset,
    load_split_piece_ids,
    summarize_examples,
)


def _encode_vlq(value: int) -> bytes:
    """Encode an integer as a MIDI variable-length quantity."""
    bytes_out = [value & 0x7F]
    value >>= 7
    while value:
        bytes_out.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(bytes_out))


def _build_minimal_midi() -> bytes:
    """Create a tiny in-memory MIDI file with two notes."""
    ticks_per_beat = 480
    track = bytearray()
    track.extend(_encode_vlq(0))
    track.extend(b"\xFF\x51\x03\x07\xA1\x20")  # 120 BPM
    track.extend(_encode_vlq(0))
    track.extend(b"\xFF\x58\x04\x04\x02\x18\x08")  # 4/4
    track.extend(_encode_vlq(0))
    track.extend(b"\xC0\x00")  # Program change
    track.extend(_encode_vlq(0))
    track.extend(b"\x90\x3C\x60")  # Note on C4 velocity 96
    track.extend(_encode_vlq(480))
    track.extend(b"\x80\x3C\x00")  # Note off
    track.extend(_encode_vlq(0))
    track.extend(b"\x90\x40\x40")  # Note on E4 velocity 64
    track.extend(_encode_vlq(240))
    track.extend(b"\x80\x40\x00")  # Note off
    track.extend(_encode_vlq(0))
    track.extend(b"\xFF\x2F\x00")  # End of track

    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big")
    header += (1).to_bytes(2, "big") + ticks_per_beat.to_bytes(2, "big")
    track_chunk = b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)
    return header + track_chunk


class MidiParserTests(unittest.TestCase):
    def test_parse_midi_bytes_extracts_notes_and_meta(self) -> None:
        piece = parse_midi_bytes(_build_minimal_midi(), piece_id="tiny")

        self.assertEqual(piece.piece_id, "tiny")
        self.assertEqual(piece.ticks_per_beat, 480)
        self.assertEqual(piece.primary_time_signature.label, "4/4")
        self.assertEqual(round(piece.primary_tempo_bpm), 120)
        self.assertEqual(len(piece.note_events), 2)
        self.assertEqual(piece.note_events[0].pitch, 60)
        self.assertEqual(piece.note_events[1].pitch, 64)
        self.assertAlmostEqual(piece.note_events[0].duration_beats, 1.0)
        self.assertAlmostEqual(piece.note_events[1].duration_beats, 0.5)


class QuantizationTests(unittest.TestCase):
    def test_quantize_piece_maps_notes_to_grid(self) -> None:
        piece = parse_midi_bytes(_build_minimal_midi(), piece_id="tiny")
        quantized = quantize_piece(piece, resolution="sixteenth")

        self.assertEqual(quantized.steps_per_beat, 4)
        self.assertEqual(quantized.bar_steps, 16)
        self.assertEqual(quantized.time_signature, "4/4")
        self.assertEqual(len(quantized.note_events), 2)
        self.assertEqual(quantized.note_events[0].start_step, 0)
        self.assertEqual(quantized.note_events[0].duration_steps, 4)
        self.assertEqual(quantized.note_events[1].start_step, 4)
        self.assertEqual(quantized.note_events[1].duration_steps, 2)
        self.assertEqual(quantized.note_events[1].position, 4)
        self.assertEqual(quantized.total_bars, 1)

    def test_segment_phrases_assigns_bar_boundaries(self) -> None:
        piece = QuantizedPiece(
            piece_id="segmentation",
            resolution="sixteenth",
            steps_per_beat=4,
            bar_steps=16,
            time_signature="4/4",
            tempo_bpm=120.0,
            note_events=[
                QuantizedEvent(
                    pitch=60,
                    velocity=90,
                    instrument=0,
                    channel=0,
                    start_step=0,
                    duration_steps=4,
                    bar=1,
                    position=0,
                    track_index=0,
                ),
                QuantizedEvent(
                    pitch=67,
                    velocity=90,
                    instrument=0,
                    channel=0,
                    start_step=128,
                    duration_steps=4,
                    bar=9,
                    position=0,
                    track_index=0,
                ),
            ],
        )

        segmented = segment_phrases(piece, strategy="bars_4")
        self.assertEqual(segmented.total_bars, 9)
        self.assertEqual(segmented.phrase_boundaries, [1, 5, 9])

    def test_segment_phrases_can_follow_cadences(self) -> None:
        piece = QuantizedPiece(
            piece_id="cadence_segmentation",
            resolution="sixteenth",
            steps_per_beat=4,
            bar_steps=16,
            time_signature="4/4",
            tempo_bpm=120.0,
            note_events=[
                QuantizedEvent(
                    pitch=60,
                    velocity=90,
                    instrument=0,
                    channel=0,
                    start_step=index * 16,
                    duration_steps=4,
                    bar=index + 1,
                    position=0,
                    track_index=0,
                    harmony="C:maj",
                )
                for index in range(9)
            ],
            metadata={"cadence_bars": [4, 8]},
            chords=["C:maj"] * 9,
        )

        segmented = segment_phrases(piece, strategy="cadence_bars_4")
        self.assertEqual(segmented.phrase_boundaries, [1, 5, 9])

    def test_annotate_quantized_piece_harmony_assigns_key_chords_and_event_labels(self) -> None:
        quantized = quantize_piece(parse_midi_bytes(_build_minimal_midi()), resolution="sixteenth")
        annotated = annotate_quantized_piece_harmony(quantized)

        self.assertNotEqual(annotated.key, "unknown")
        self.assertTrue(annotated.chords)
        self.assertTrue(all(event.harmony != "unknown" for event in annotated.note_events))
        self.assertIn("cadence_bars", annotated.metadata)


class BlockEncodingTests(unittest.TestCase):
    def test_encode_piece_to_blocks_buckets_core_features(self) -> None:
        parsed = ParsedPiece(
            piece_id="manual",
            ticks_per_beat=480,
            note_events=[
                NoteEvent(
                    pitch=60,
                    velocity=96,
                    start_tick=0,
                    end_tick=480,
                    start_beat=0.0,
                    duration_beats=1.0,
                    instrument=0,
                    channel=0,
                    track_index=0,
                ),
                NoteEvent(
                    pitch=64,
                    velocity=64,
                    start_tick=480,
                    end_tick=720,
                    start_beat=1.0,
                    duration_beats=0.5,
                    instrument=0,
                    channel=0,
                    track_index=0,
                ),
            ],
        )

        quantized = annotate_quantized_piece_harmony(quantize_piece(parsed))
        quantized = segment_phrases(quantized, strategy="single_phrase")
        example = encode_piece_to_blocks(quantized, duration_bins=8, velocity_bins=4)

        self.assertEqual(len(example), 2)
        self.assertEqual(example.event_blocks[0].pitch, 60)
        self.assertEqual(example.event_blocks[0].duration, 3)
        self.assertEqual(example.event_blocks[0].velocity, 3)
        self.assertEqual(example.event_blocks[0].bar, 1)
        self.assertEqual(example.event_blocks[0].phrase_index, 0)
        self.assertEqual(example.event_blocks[1].duration, 1)
        self.assertEqual(example.event_blocks[1].velocity, 2)
        self.assertEqual(example.event_blocks[0].phrase_flag, 3)
        self.assertEqual(example.event_blocks[1].phrase_flag, 3)
        self.assertEqual(example.metadata["harmony_map"]["unknown"], 0)
        self.assertIn("C:maj", example.metadata["harmony_map"])
        self.assertEqual(example.event_blocks[0].harmony, example.metadata["harmony_map"]["C:maj"])

    def test_encode_piece_to_blocks_buckets_bar_positions_into_configured_vocab(self) -> None:
        piece = QuantizedPiece(
            piece_id="compound_meter",
            resolution="sixteenth",
            steps_per_beat=4,
            bar_steps=24,
            time_signature="6/8",
            tempo_bpm=120.0,
            note_events=[
                QuantizedEvent(
                    pitch=72,
                    velocity=90,
                    instrument=0,
                    channel=0,
                    start_step=23,
                    duration_steps=2,
                    bar=1,
                    position=23,
                    track_index=0,
                )
            ],
            phrase_boundaries=[1],
        )

        example = encode_piece_to_blocks(
            piece,
            duration_bins=8,
            velocity_bins=4,
            bar_position_bins=16,
        )

        self.assertEqual(example.event_blocks[0].bar_position, 15)


class MidiRenderingTests(unittest.TestCase):
    def test_render_round_trip_preserves_quantized_note_grid(self) -> None:
        original = segment_phrases(quantize_piece(parse_midi_bytes(_build_minimal_midi())), "single_phrase")
        rendered_bytes = render_piece_to_midi_bytes(original)
        reparsed = parse_midi_bytes(rendered_bytes, piece_id="roundtrip")
        requantized = quantize_piece(reparsed, resolution="sixteenth")

        roundtrip_summary = [
            (event.pitch, event.start_step, event.duration_steps)
            for event in requantized.note_events
        ]
        original_summary = [
            (event.pitch, event.start_step, event.duration_steps)
            for event in original.note_events
        ]
        self.assertEqual(roundtrip_summary, original_summary)


class DatasetPreparationTests(unittest.TestCase):
    def test_prepare_dataset_writes_json_artifacts_and_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            splits_dir = root / "splits"
            nested_dir = raw_dir / "nested collection"
            nested_dir.mkdir(parents=True)

            for relative_path in (
                raw_dir / "alpha.mid",
                raw_dir / "beta.midi",
                nested_dir / "gamma piece.mid",
                nested_dir / "delta-piece.mid",
            ):
                relative_path.write_bytes(_build_minimal_midi())

            result = prepare_dataset(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                splits_dir=splits_dir,
                annotate_harmony=True,
                train_ratio=0.5,
                val_ratio=0.25,
                seed=7,
            )

            self.assertEqual(len(result.artifacts), 4)
            self.assertEqual(len(result.failures), 0)
            self.assertEqual(len(result.split_assignments["train"]), 2)
            self.assertEqual(len(result.split_assignments["val"]), 1)
            self.assertEqual(len(result.split_assignments["test"]), 1)

            manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["piece_count"], 4)
            self.assertEqual(manifest["failure_count"], 0)
            self.assertTrue(manifest["annotate_harmony"])
            self.assertEqual(manifest["dataset_kind"], "generic")
            self.assertEqual(manifest["dataset_source_mode"], "performance")

            sample_artifact = processed_dir / "nested_collection__gamma_piece.json"
            self.assertTrue(sample_artifact.exists())
            loaded_piece = load_quantized_piece_json(sample_artifact)
            self.assertEqual(loaded_piece.piece_id, "nested_collection__gamma_piece")
            self.assertEqual(loaded_piece.source_path, "nested collection/gamma piece.mid")
            self.assertEqual(len(loaded_piece.note_events), 2)
            self.assertEqual(loaded_piece.phrase_boundaries, [1])
            self.assertNotEqual(loaded_piece.key, "unknown")
            self.assertTrue(loaded_piece.chords)
            self.assertTrue(all(event.harmony != "unknown" for event in loaded_piece.note_events))

            all_split_ids = set()
            for split_name in ("train", "val", "test"):
                payload = json.loads((splits_dir / f"{split_name}.json").read_text(encoding="utf-8"))
                self.assertEqual(payload["split"], split_name)
                all_split_ids.update(payload["piece_ids"])
            self.assertEqual(
                all_split_ids,
                {artifact.piece_id for artifact in result.artifacts},
            )

    def test_toy_data_bootstrap_flows_into_processed_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            splits_dir = root / "splits"

            written = generate_toy_dataset(raw_dir, num_pieces=5, seed=3)
            self.assertEqual(len(written), 5)

            prepare_dataset(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                splits_dir=splits_dir,
                train_ratio=0.6,
                val_ratio=0.2,
                seed=11,
            )
            train_examples = load_processed_dataset(
                processed_dir=processed_dir,
                splits_dir=splits_dir,
                split="train",
                duration_bins=8,
                velocity_bins=4,
            )
            summary = summarize_examples(train_examples, split="train")

            self.assertGreater(summary.piece_count, 0)
            self.assertGreater(summary.event_block_count, 0)
            self.assertGreater(summary.mean_length, 0)
            self.assertEqual(len(load_split_piece_ids(splits_dir, split="train")), 3)

    def test_prepare_dataset_uses_official_maestro_splits_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            splits_dir = root / "splits"
            maestro_dir = raw_dir / "maestro-v3.0.0"
            (maestro_dir / "2018").mkdir(parents=True)
            (maestro_dir / "2019").mkdir(parents=True)
            (maestro_dir / "2020").mkdir(parents=True)

            midi_paths = [
                maestro_dir / "2018" / "alpha piece.midi",
                maestro_dir / "2019" / "beta-piece.midi",
                maestro_dir / "2020" / "gamma_piece.midi",
            ]
            for midi_path in midi_paths:
                midi_path.write_bytes(_build_minimal_midi())

            metadata_csv = maestro_dir / "maestro-v3.0.0.csv"
            metadata_csv.write_text(
                "\n".join(
                    [
                        "canonical_composer,canonical_title,split,year,midi_filename,audio_filename,duration",
                        "Composer A,Title A,train,2018,2018/alpha piece.midi,2018/alpha.wav,1.0",
                        "Composer B,Title B,validation,2019,2019/beta-piece.midi,2019/beta.wav,1.0",
                        "Composer C,Title C,test,2020,2020/gamma_piece.midi,2020/gamma.wav,1.0",
                    ]
                ),
                encoding="utf-8",
            )

            detected = detect_maestro_official_splits(raw_dir)
            self.assertEqual(len(detected["train"]), 1)
            self.assertEqual(len(detected["val"]), 1)
            self.assertEqual(len(detected["test"]), 1)

            result = prepare_dataset(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                splits_dir=splits_dir,
                train_ratio=0.1,
                val_ratio=0.1,
                seed=999,
            )

            self.assertEqual(result.split_strategy, "official_maestro")
            self.assertEqual(
                result.split_assignments,
                {
                    "train": ["maestro_v3_0_0__2018__alpha_piece"],
                    "val": ["maestro_v3_0_0__2019__beta_piece"],
                    "test": ["maestro_v3_0_0__2020__gamma_piece"],
                },
            )

            manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["split_strategy"], "official_maestro")

    def test_prepare_dataset_supports_asap_score_mode_with_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            splits_dir = root / "splits"

            for relative_path in (
                Path("Bach/Prelude_01/score.mid"),
                Path("Bach/Prelude_01/perf_1.mid"),
                Path("Bach/Prelude_01/perf_2.mid"),
                Path("Bach/Prelude_02/score.mid"),
                Path("Bach/Prelude_02/perf_1.mid"),
            ):
                output_path = raw_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(_build_minimal_midi())

            (raw_dir / "metadata.csv").write_text(
                "\n".join(
                    [
                        "composer,title,folder,xml_score,midi_score,midi_performance",
                        "Bach,Prelude 1,Bach/Prelude_01,Bach/Prelude_01/score.xml,Bach/Prelude_01/score.mid,Bach/Prelude_01/perf_1.mid",
                        "Bach,Prelude 1,Bach/Prelude_01,Bach/Prelude_01/score.xml,Bach/Prelude_01/score.mid,Bach/Prelude_01/perf_2.mid",
                        "Bach,Prelude 2,Bach/Prelude_02,Bach/Prelude_02/score.xml,Bach/Prelude_02/score.mid,Bach/Prelude_02/perf_1.mid",
                    ]
                ),
                encoding="utf-8",
            )
            (raw_dir / "asap_annotations.json").write_text(
                json.dumps(
                    {
                        "Bach/Prelude_01/perf_1.mid": {
                            "midi_score_time_signatures": {"0.0": ["3/4", 3, 4]},
                            "midi_score_key_signatures": {"0.0": [0, 0]},
                            "midi_score_beats": [0.0, 1.0, 2.0],
                            "midi_score_downbeats": [0.0],
                            "downbeats_score_map": [1],
                            "score_and_performance_aligned": True,
                        },
                        "Bach/Prelude_01/perf_2.mid": {
                            "midi_score_time_signatures": {"0.0": ["4/4", 4, 4]},
                            "score_and_performance_aligned": False,
                        },
                        "Bach/Prelude_02/perf_1.mid": {
                            "midi_score_time_signatures": {"0.0": ["2/4", 2, 4]},
                            "midi_score_key_signatures": {"0.0": [7, 1]},
                            "score_and_performance_aligned": True,
                        },
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            self.assertTrue(detect_asap_dataset(raw_dir))
            self.assertEqual(detect_dataset_kind(raw_dir), "asap")

            result = prepare_dataset(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                splits_dir=splits_dir,
                dataset_kind="asap",
                dataset_source_mode="score",
                use_dataset_annotations=True,
            )

            self.assertEqual(len(result.artifacts), 2)
            self.assertEqual(result.split_strategy, "grouped_asap_piece")

            piece = load_quantized_piece_json(processed_dir / "bach__prelude_01__score.json")
            self.assertEqual(piece.time_signature, "3/4")
            self.assertEqual(piece.metadata["dataset_kind"], "asap")
            self.assertEqual(piece.metadata["dataset_source_mode"], "score")
            self.assertEqual(piece.metadata["composer"], "Bach")
            self.assertEqual(piece.metadata["title"], "Prelude 1")
            self.assertTrue(piece.metadata["score_and_performance_aligned"])

            manifest = json.loads((processed_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset_kind"], "asap")
            self.assertEqual(manifest["dataset_source_mode"], "score")
            self.assertTrue(manifest["use_dataset_annotations"])

    def test_prepare_dataset_groups_asap_performances_in_same_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            splits_dir = root / "splits"

            for relative_path in (
                Path("Mozart/Sonata_01/score.mid"),
                Path("Mozart/Sonata_01/perf_1.mid"),
                Path("Mozart/Sonata_01/perf_2.mid"),
                Path("Mozart/Sonata_02/score.mid"),
                Path("Mozart/Sonata_02/perf_1.mid"),
            ):
                output_path = raw_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(_build_minimal_midi())

            (raw_dir / "metadata.csv").write_text(
                "\n".join(
                    [
                        "composer,title,folder,xml_score,midi_score,midi_performance",
                        "Mozart,Sonata 1,Mozart/Sonata_01,Mozart/Sonata_01/score.xml,Mozart/Sonata_01/score.mid,Mozart/Sonata_01/perf_1.mid",
                        "Mozart,Sonata 1,Mozart/Sonata_01,Mozart/Sonata_01/score.xml,Mozart/Sonata_01/score.mid,Mozart/Sonata_01/perf_2.mid",
                        "Mozart,Sonata 2,Mozart/Sonata_02,Mozart/Sonata_02/score.xml,Mozart/Sonata_02/score.mid,Mozart/Sonata_02/perf_1.mid",
                    ]
                ),
                encoding="utf-8",
            )

            result = prepare_dataset(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                splits_dir=splits_dir,
                dataset_kind="asap",
                dataset_source_mode="performance",
                train_ratio=0.5,
                val_ratio=0.0,
                seed=7,
            )

            self.assertEqual(len(result.artifacts), 3)
            self.assertEqual(result.split_strategy, "grouped_asap_piece")

            split_payloads = {
                split_name: json.loads((splits_dir / f"{split_name}.json").read_text(encoding="utf-8"))["piece_ids"]
                for split_name in ("train", "val", "test")
            }
            perf_one = "mozart__sonata_01__perf_1"
            perf_two = "mozart__sonata_01__perf_2"
            split_names = [
                split_name
                for split_name, piece_ids in split_payloads.items()
                if perf_one in piece_ids or perf_two in piece_ids
            ]
            self.assertEqual(len(split_names), 1)
            self.assertIn(perf_one, split_payloads[split_names[0]])
            self.assertIn(perf_two, split_payloads[split_names[0]])


class TrainingDatasetTests(unittest.TestCase):
    def test_autoregressive_dataset_and_collate_build_padded_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            splits_dir = root / "splits"

            generate_toy_dataset(raw_dir, num_pieces=4, seed=9)
            prepare_dataset(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                splits_dir=splits_dir,
                train_ratio=0.5,
                val_ratio=0.25,
                seed=3,
            )

            dataset = AutoregressiveTokenDataset(
                processed_dir=processed_dir,
                splits_dir=splits_dir,
                split="train",
                duration_bins=8,
                velocity_bins=4,
                cache_examples=True,
            )

            self.assertEqual(len(dataset), 2)
            sample_a = dataset[0]
            sample_b = dataset[1]
            self.assertGreater(sample_a["sequence_length"], 0)
            self.assertEqual(sample_a["inputs"]["pitch"].shape[0], sample_a["sequence_length"])
            self.assertEqual(sample_a["targets"]["pitch"].shape[0], sample_a["sequence_length"])

            batch = collate_autoregressive_batch([sample_a, sample_b])
            self.assertEqual(batch.inputs["pitch"].shape[0], 2)
            self.assertEqual(batch.targets["pitch"].shape, batch.inputs["pitch"].shape)
            self.assertEqual(batch.attention_mask.shape, batch.inputs["pitch"].shape)
            self.assertTrue(batch.attention_mask[0, : sample_a["sequence_length"]].all().item())
            self.assertEqual(batch.lengths.tolist(), [sample_a["sequence_length"], sample_b["sequence_length"]])
            self.assertEqual(batch.phrase_ids.shape, batch.inputs["pitch"].shape)
            self.assertEqual(batch.phrase_mask.shape[0], 2)
            self.assertIn("recurrence", batch.conductor_targets)
            self.assertIn("harmonic_zone", batch.conductor_targets)

    def test_autoregressive_dataloader_returns_batch_object(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            processed_dir = root / "processed"
            splits_dir = root / "splits"

            generate_toy_dataset(raw_dir, num_pieces=5, seed=4)
            prepare_dataset(
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                splits_dir=splits_dir,
                train_ratio=0.6,
                val_ratio=0.2,
                seed=8,
            )

            dataset = AutoregressiveTokenDataset(
                processed_dir=processed_dir,
                splits_dir=splits_dir,
                split="train",
                duration_bins=8,
                velocity_bins=4,
                limit=3,
            )
            loader = create_autoregressive_dataloader(dataset, batch_size=2)
            batch = next(iter(loader))

            self.assertEqual(batch.inputs["pitch"].shape[0], 2)
            self.assertEqual(batch.targets["duration"].shape[0], 2)
            self.assertEqual(batch.lengths.shape[0], 2)
            self.assertEqual(len(batch.piece_ids), 2)
            self.assertTrue(all(length > 0 for length in batch.lengths.tolist()))


if __name__ == "__main__":
    unittest.main()
