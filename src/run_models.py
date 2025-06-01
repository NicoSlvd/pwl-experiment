from main import main

for dataset in ["LPMC"]:
    for model_type in ["linear", "cubic", "constant_linear", "constant_cubic"]:
        for monotone in [True, False]:
            for model in ["RUMBoost", "APLR"]:
                if model == "APLR" and model_type != "linear":
                    continue
                if model == "RUMBoost" and model_type == "constant":
                    continue
                main([
                    "--model", model,
                    "--model_type", model_type,
                    "--monotone", str(monotone).lower(),
                    "--dataset", dataset,
                ])