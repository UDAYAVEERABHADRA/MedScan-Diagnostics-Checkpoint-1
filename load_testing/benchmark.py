"""Run a reproducible 50-concurrent-client API benchmark. Start the API first."""
import asyncio, argparse, statistics, time
import httpx

PAYLOAD = {
    "Age": 35, "Gender": "F", "Fever": 1, "Cough": 1, "Fatigue": 0,
    "Breathlessness": 0, "Chest_Pain": 0, "Nausea": 1, "Headache": 1,
    "BMI": 24.5, "BP_Systolic": 118, "BP_Diastolic": 78, "SpO2": 98,
    "Glucose_mg_dL": 95, "HbA1c": 5.4
}

async def main(base_url, requests, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    latencies, failures = [], 0
    async with httpx.AsyncClient(timeout=30) as client:
        async def one():
            nonlocal failures
            async with semaphore:
                start = time.perf_counter()
                try:
                    response = await client.post(base_url.rstrip('/') + '/predict', json=PAYLOAD)
                    elapsed = (time.perf_counter() - start) * 1000
                    if response.status_code == 200:
                        latencies.append(elapsed)
                    else:
                        failures += 1
                except Exception:
                    failures += 1
        wall_start = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(requests)))
        wall = time.perf_counter() - wall_start
    if not latencies:
        raise SystemExit("All requests failed. Check that the API is running and the model is loaded.")
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered)-1, int(0.95*len(ordered)))]
    print(f"Total requests: {requests}")
    print(f"Concurrency: {concurrency}")
    print(f"Successful: {len(latencies)}")
    print(f"Failed: {failures}")
    print(f"Total wall time: {wall:.3f} s")
    print(f"Throughput: {len(latencies)/wall:.2f} requests/second")
    print(f"Mean latency: {statistics.mean(latencies):.2f} ms")
    print(f"Median latency: {statistics.median(latencies):.2f} ms")
    print(f"p95 latency: {p95:.2f} ms")
    print(f"Min/Max latency: {min(latencies):.2f}/{max(latencies):.2f} ms")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--requests', type=int, default=250)
    parser.add_argument('--concurrency', type=int, default=50)
    args = parser.parse_args()
    asyncio.run(main(args.url, args.requests, args.concurrency))
