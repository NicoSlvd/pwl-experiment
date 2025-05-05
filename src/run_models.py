from main import main

for model_type in ["linear", "cubic", "constant_linear", "constant_cubic"]:
    for monotone in [True, False]:
        for dataset in ["LPMC", "SwissMetro"]:
            for model in ["RUMBoost", "APLR"]:
                if model == "APLR" and model_type != "linear":
                    continue
                if model == "RUMBoost" and dataset == "SwissMetro" and monotone and model_type == "constant":
                    continue
                if model == "RUMBoost" and dataset == "LPMC" and monotone and model_type == "constant":
                    continue
                main([
                    "--model", model,
                    "--model_type", model_type,
                    "--monotone", str(monotone).lower(),
                    "--dataset", dataset,
                ])