import argparse
from flask import Flask, request, jsonify

from tasks import DataExample, BaseTask, TASKS
from train.evaluator import TaskEvaluatorBuilder, TaskEvaluator


class ModelHandler(object):

    def __init__(self, task: str, arch: str, model_dir: str):
        assert arch in ("roberta_base", "roberta_large", "bart_base", "bart_large")
        self.task: BaseTask = TASKS.get(task)()
        self.arch = arch
        self.model_dir = model_dir
        self.model: TaskEvaluator = self.__create_evaluator()

    def __create_evaluator(self) -> TaskEvaluator:
        builder = TaskEvaluatorBuilder(self.task, self.arch, self.model_dir)
        return builder.build()

    def predict(self, example: DataExample):
        return self.model.predict(example)


def create_app():
    app = Flask(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    handler = ModelHandler(args.task, args.arch, args.model_dir)

    @app.route("/predict", methods=["POST"])
    def predict():
        data: any = request.json
        inputs = data.get("inputs")
        inputs = [inputs] if isinstance(inputs, str) else inputs
        output = handler.predict(DataExample(inputs, None))
        return jsonify({"inputs": inputs, "output": output})

    @app.route("/", methods=["POST", "GET"])
    def index():
        res = handler.task.spec().to_json()
        res["name"] = args.task
        return res

    return app, args


if __name__ == '__main__':
    app, args = create_app()
    app.run(host=args.host, port=args.port, threaded=False)
