from main import main

for model_type in ["linear"]:#, "constant_linear"]: #"constant","linear", , "constant_cubic", "linear", "cubic", 
    for dataset in ["LPMC"]:#, "SwissMetro"]:
        for monotone in [True]:#, False]:
            for model in ["RUMBoost"]:#"APLR"
                if model == "APLR" and model_type != "linear":
                    continue
                main([
                    "--model", model,
                    "--model_type", model_type,
                    "--monotone", str(monotone).lower(),
                    "--dataset", dataset,
                    "--save_model", "true",
                    "--all_boosters", "true",
                    "--max_bin", "10"
                ])