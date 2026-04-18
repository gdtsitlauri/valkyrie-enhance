import json

from valkyrie.config import ValkyrieConfig, dump_config, load_config
from valkyrie.experiments import export_benchmark_rows, run_module_ablation, run_synthetic_benchmarks
from valkyrie.reporting import write_scenario_summary


def test_config_roundtrip(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    config = ValkyrieConfig()
    config.quality.scale_factor = 1.5
    dump_config(config, path)
    loaded = load_config(path)
    assert loaded.quality.scale_factor == 1.5


def test_benchmark_export_and_summary(tmp_path) -> None:
    rows = run_synthetic_benchmarks(ValkyrieConfig(), seeds=(42,))
    json_path = tmp_path / "rows.json"
    md_path = tmp_path / "summary.md"
    export_benchmark_rows(rows, json_path)
    write_scenario_summary(rows, md_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))
    assert "Synthetic Benchmark Summary" in md_path.read_text(encoding="utf-8")


def test_module_ablation_outputs_full_pipeline_key() -> None:
    ablation = run_module_ablation(ValkyrieConfig())
    assert "full_pipeline_psnr" in ablation
