from main import main


for model_type in ["constant","constant_linear"]: #"constant_cubic", "linear", "cubic", 
    for dataset in ["LPMC", "SwissMetro"]:
        for monotone in [True, False]:
            for model in ["RUMBoost"]:
                if model == "APLR" and model_type != "linear":
                    continue
                main([
                    "--model", model,
                    "--model_type", model_type,
                    "--monotone", str(monotone).lower(),
                    "--dataset", dataset,
                    "--save_model", "true"
                ])