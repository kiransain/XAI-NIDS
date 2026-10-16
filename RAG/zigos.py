
import json
from collections import Counter, defaultdict
from pathlib import Path

from experiment_set import EXPERIMENT_SET


RESULTS_FILE = Path("paper_results_llm_as_judge.json")
EXPECTED_ENGINES = {"none", "bm25", "vector"}


def main():
    with RESULTS_FILE.open("r", encoding="utf-8") as f:
        results = json.load(f)

    # Expected query IDs from the experiment set
    expected_ids = [experiment["id"] for experiment in EXPERIMENT_SET]
    expected_id_set = set(expected_ids)

    # Detect duplicate IDs in the experiment set
    expected_counts = Counter(expected_ids)
    duplicate_experiment_ids = sorted(
        query_id
        for query_id, count in expected_counts.items()
        if count > 1
    )

    # Collect engines found for each query ID
    engines_by_query = defaultdict(list)
    unexpected_engines = []
    missing_fields = []

    for index, result in enumerate(results):
        query_id = result.get("query_id")
        engine = result.get("engine")

        if not query_id:
            missing_fields.append(
                f"Result at JSON index {index} has no query_id"
            )
            continue

        if not engine:
            missing_fields.append(
                f"Result at JSON index {index} ({query_id}) has no engine"
            )
            continue

        engines_by_query[query_id].append(engine)

        if engine not in EXPECTED_ENGINES:
            unexpected_engines.append((query_id, engine))

    result_id_set = set(engines_by_query)

    # 1. Query IDs with no results whatsoever
    missing_query_ids = sorted(expected_id_set - result_id_set)

    # 2. Expected queries with missing engines
    missing_engine_results = {}
    for query_id in sorted(expected_id_set):
        found = Counter(engines_by_query.get(query_id, []))
        missing = sorted(
            engine
            for engine in EXPECTED_ENGINES
            if found[engine] == 0
        )
        if missing:
            missing_engine_results[query_id] = missing

    # 3. Query IDs in JSON that are not in the experiment set
    unexpected_query_ids = sorted(result_id_set - expected_id_set)

    # 4. Duplicate query_id + engine combinations
    duplicate_results = []
    for query_id, engines in sorted(engines_by_query.items()):
        counts = Counter(engines)
        for engine, count in sorted(counts.items()):
            if count > 1:
                duplicate_results.append(
                    (query_id, engine, count)
                )

    # 5. Queries with exactly the expected three engines,
    #    with each engine appearing exactly once
    complete_query_ids = [
        query_id
        for query_id in expected_id_set
        if Counter(engines_by_query.get(query_id, []))
        == Counter({"none": 1, "bm25": 1, "vector": 1})
    ]

    print("=" * 65)
    print("QUERY COVERAGE CHECK")
    print("=" * 65)
    print(f"Experiments defined:          {len(expected_ids)}")
    print(f"Unique experiment IDs:        {len(expected_id_set)}")
    print(f"Judge result rows:             {len(results)}")
    print(f"Query IDs found in JSON:       {len(result_id_set)}")
    print(f"Complete queries (3 engines):  {len(complete_query_ids)}")
    print(f"Missing query IDs entirely:    {len(missing_query_ids)}")
    print(f"Queries missing engines:       {len(missing_engine_results)}")
    print(f"Unexpected query IDs:          {len(unexpected_query_ids)}")
    print(f"Duplicate query-engine pairs:  {len(duplicate_results)}")
    print(f"Duplicate experiment IDs:      {len(duplicate_experiment_ids)}")
    print(f"Malformed result rows:         {len(missing_fields)}")
    print(f"Unexpected engine entries:     {len(unexpected_engines)}")

    if missing_query_ids:
        print("\n[MISSING QUERY IDs — NO RESULTS]")
        for query_id in missing_query_ids:
            print(f"  {query_id}")

    if missing_engine_results:
        print("\n[MISSING ENGINE RESULTS]")
        for query_id, missing in missing_engine_results.items():
            found = sorted(set(engines_by_query.get(query_id, [])))
            print(
                f"  {query_id}: missing {', '.join(missing)} "
                f"(found: {', '.join(found) or 'none'})"
            )

    if unexpected_query_ids:
        print("\n[UNEXPECTED QUERY IDs — NOT IN EXPERIMENT_SET]")
        for query_id in unexpected_query_ids:
            print(f"  {query_id}")

    if duplicate_results:
        print("\n[DUPLICATE QUERY-ENGINE RESULTS]")
        for query_id, engine, count in duplicate_results:
            print(f"  {query_id} / {engine}: appears {count} times")

    if duplicate_experiment_ids:
        print("\n[DUPLICATE IDS IN EXPERIMENT_SET]")
        for query_id in duplicate_experiment_ids:
            print(f"  {query_id}")

    if missing_fields:
        print("\n[MALFORMED RESULT ROWS]")
        for message in missing_fields:
            print(f"  {message}")

    if unexpected_engines:
        print("\n[UNEXPECTED ENGINES]")
        for query_id, engine in unexpected_engines:
            print(f"  {query_id}: {engine}")

    # Overall pass/fail
    passed = (
        not missing_query_ids
        and not missing_engine_results
        and not unexpected_query_ids
        and not duplicate_results
        and not duplicate_experiment_ids
        and not missing_fields
        and not unexpected_engines
    )

    print("\n" + "=" * 65)
    print("RESULT:", "PASS — all queries have exactly 3 engine results"
          if passed else "FAIL — coverage problems found")
    print("=" * 65)


if __name__ == "__main__":
    main()
