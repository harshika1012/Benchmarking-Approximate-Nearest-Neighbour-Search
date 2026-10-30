#!/usr/bin/env bash
# Regenerate every number, table and figure in the report.
#   bash run_all.sh          full datasets (several hours, single thread)
#   bash run_all.sh --quick  first 100K vectors, smaller grids (< 15 min)
set -euo pipefail
cd "$(dirname "$0")"

QUICK=""
[[ "${1:-}" == "--quick" ]] && QUICK="--quick"
ROOT=results; [[ -n $QUICK ]] && ROOT=results_quick

export PYTHONPATH="$PWD/code"
# One thread everywhere, including the BLAS used by batch search.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for f in data/sift/sift-128-euclidean.hdf5 data/glove/glove-100-angular.hdf5; do
  [[ -f $f ]] || { echo "Missing $f (see README: Data)"; exit 1; }
done

echo "== Part 1: recall tests and exact baseline"
python code/test_recall.py
for ds in sift glove; do
  python code/exact_baseline.py --dataset "$ds" $QUICK
done

echo "== Part 2: HNSW, IVF-Flat, IVF-PQ"
for ds in sift glove; do
  for ix in hnsw ivfflat ivfpq; do
    python code/benchmark.py --dataset "$ds" --index "$ix" $QUICK
  done
done
python code/report_part2.py $QUICK

echo "== Part 3: scaling on SIFT"
if [[ -n $QUICK ]]; then SIZES="10000 30000"; LARGEST=100000
else SIZES="100000 300000"; LARGEST=1000000; fi
for n in $SIZES; do
  for ix in exact ivfflat hnsw; do
    python code/benchmark.py --dataset sift --index "$ix" --n-base "$n" \
      --out-dir "$ROOT/part3" $QUICK
  done
done
# HNSW and IVF-Flat at the largest size are the Part 2 runs.
python code/benchmark.py --dataset sift --index exact --n-base "$LARGEST" \
  --out-dir "$ROOT/part3" $QUICK
python code/report_part3.py $QUICK

echo "== Part 4: drift on synthetic data"
python code/drift_experiment.py $QUICK
python code/report_part4.py $QUICK

echo "== Combine results"
python code/collect_results.py $QUICK
echo "Done: tables in $ROOT/tables, figures in $ROOT/figures, all runs in $ROOT/results.csv"
