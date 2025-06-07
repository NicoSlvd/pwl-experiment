from main import main

for model_type in ["linear"]:#, "constant_linear"]: #"constant","linear", , "constant_cubic", "linear", "cubic", 
    for dataset in ["SwissMetro"]:#, "LPMC"]:
        for monotone in [True]: #, False]:#, False]:
            for model in ["RUMBoost"]:#"APLR"
                if model == "APLR" and model_type != "linear":
                    continue
                # if model_type == "linear" and dataset == "SwissMetro":
                #     continue
                # if model_type == "constant_linear" and monotone == False:
                #     continue
                main([
                    "--model", model,
                    "--model_type", model_type,
                    "--monotone", str(monotone).lower(),
                    "--dataset", dataset,
                    "--save_model", "true",
                    "--all_boosters", "true",
                    "--max_bin", "10",
                ])