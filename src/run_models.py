from main import main

for dataset in ["SwissMetro"]:#, "LPMC"]:
    for model_type in ["constant","constant_linear"]:#, "linear"]: #"constant_cubic", "linear", "cubic", 
        for monotone in [True, False]:
            for model in ["RUMBoost"]:#"APLR"
                if model == "APLR" and model_type != "linear":
                    continue
                # if model_type == "constant":
                #     continue
                # if model_type == "constant_linear" and monotone == False:
                #     continue
                main([
                    "--model", model,
                    "--model_type", model_type,
                    "--monotone", str(monotone).lower(),
                    "--dataset", dataset,
                    "--save_model", "true",
                    "--all_boosters", "true"
                ])