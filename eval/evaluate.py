import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline import build_pipeline, run_pipeline

# ── 10 test questions with expected keywords ──────────
test_cases = [
    {
        "question": "What is Machine Learning?",
        "expected_keywords": ["artificial intelligence", "data", "learn", "predictions"]
    },
    {
        "question": "What is supervised learning?",
        "expected_keywords": ["supervised", "labeled", "training"]
    },
    {
        "question": "What is overfitting?",
        "expected_keywords": ["memorize", "training", "overfitting"]
    },
    {
        "question": "What is the difference between regression and classification?",
        "expected_keywords": ["continuous", "categories", "regression", "classification"]
    },
    {
        "question": "What is Random Forest?",
        "expected_keywords": ["random forest", "algorithm", "tree"]
    },
    {
        "question": "What is cross validation?",
        "expected_keywords": ["cross", "validation", "split"]
    },
    {
        "question": "What is feature engineering?",
        "expected_keywords": ["feature", "engineering"]
    },
    {
        "question": "What is normalization?",
        "expected_keywords": ["normalization", "scale", "range"]
    },
    {
        "question": "What is gradient descent?",
        "expected_keywords": ["gradient", "descent", "optimize"]
    },
    {
        "question": "What is the capital of France?",
        "expected_keywords": ["paris"]  # should fail - not in docs
    },
]


def evaluate_answer(answer, expected_keywords):
    """
    Simple keyword-based evaluation.
    Checks how many expected keywords appear in the answer.
    Returns a score between 0 and 1.
    """
    if not answer:
        return 0.0
    answer_lower = answer.lower()
    matched = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return round(matched / len(expected_keywords), 2)


def run_evaluation():
    print("="*60)
    print("PROJECT 13 - EVALUATION REPORT")
    print("="*60)

    # Build pipeline once
    print("\nBuilding pipeline...")
    collection = build_pipeline("data")
    print("Pipeline ready!\n")

    results = []
    total_score = 0
    safe_count = 0
    entailment_count = 0

    for i, test in enumerate(test_cases):
        question = test["question"]
        expected = test["expected_keywords"]

        print(f"[{i+1}/10] {question}")

        # Run pipeline
        result = run_pipeline(question, collection)

        # Evaluate answer quality
        keyword_score = evaluate_answer(result["answer"], expected)
        total_score += keyword_score

        if result["is_safe"]:
            safe_count += 1
        if result["hallucination_check"] == "entailment":
            entailment_count += 1

        results.append({
            "question": question,
            "answer": result["answer"][:100] + "..." if len(result["answer"]) > 100 else result["answer"],
            "is_safe": result["is_safe"],
            "confidence": round(result["confidence_score"], 2),
            "hallucination": result["hallucination_check"],
            "keyword_score": keyword_score
        })

        print(f"  Answer: {result['answer'][:80]}...")
        print(f"  Safe: {result['is_safe']} | Confidence: {result['confidence_score']:.2f} | Keyword Score: {keyword_score}")
        print()

    # Summary
    avg_score = round(total_score / len(test_cases), 2)
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total questions:     {len(test_cases)}")
    print(f"Safe answers:        {safe_count}/{len(test_cases)}")
    print(f"Entailment answers:  {entailment_count}/{len(test_cases)}")
    print(f"Avg keyword score:   {avg_score}")
    print(f"Hallucination rate:  {round((len(test_cases) - entailment_count) / len(test_cases) * 100, 1)}%")
    print("="*60)

    # Save results to file
    save_results(results, avg_score, safe_count, entailment_count)
    print("\nResults saved to eval/results.txt")


def save_results(results, avg_score, safe_count, entailment_count):
    with open("eval/results.txt", "w") as f:
        f.write("PROJECT 13 - EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")

        for i, r in enumerate(results):
            f.write(f"Q{i+1}: {r['question']}\n")
            f.write(f"Answer: {r['answer']}\n")
            f.write(f"Safe: {r['is_safe']} | Confidence: {r['confidence']} | ")
            f.write(f"Hallucination: {r['hallucination']} | Keyword Score: {r['keyword_score']}\n")
            f.write("-"*40 + "\n")

        f.write(f"\nSUMMARY\n")
        f.write(f"Safe answers: {safe_count}/10\n")
        f.write(f"Entailment: {entailment_count}/10\n")
        f.write(f"Avg keyword score: {avg_score}\n")


if __name__ == "__main__":
    run_evaluation()