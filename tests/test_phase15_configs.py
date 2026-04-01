from scripts.generate_trackB_phase15_configs import build_phase15_configs


def test_phase15_config_grid():
    configs = build_phase15_configs(seeds=list(range(2)))
    # 2 seeds x 11 lambda2 values x 1 decoy level
    assert len(configs) == 22
    sample = configs[0][1]
    assert sample["track"] == "B"
    assert sample["phase"] == 1.5
    assert sample["graph_family"] == "chain"
    assert sample["support_type"] == "connected_disjoint"
    assert sample["decoy_intensity"] == "high"
