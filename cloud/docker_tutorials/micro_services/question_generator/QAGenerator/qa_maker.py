from QAGenerator.question_generation.pipelines import pipeline


def qa_retriever_load(model_path):
    return pipeline("e2e-qg", model=model_path)
