from transformers import TrainerCallback


log_json = {"metrics": [], "args": {}}


class Logger(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        log_json["metrics"].append(metrics)

        if not log_json["args"]:
            args_dict = args.to_dict()
            log_json["args"] = args_dict
