#!/bin/bash
# Assemble the 3DGS source trees for the ScanNet++ sweep, one per (sequence, arm):
#
#   3dgs_data_snpp/dslr/<seq>_<arm>/
#       images        -> datasets/scannetpp/data/amb3r/<seq>/dslr/images   (shared by all arms)
#       sparse/0/cameras.bin   -> scannetpp_amb3r/<seq>/dslr/sfm_<arm>/0/cameras.bin
#       sparse/0/images.bin    -> ...
#       sparse/0/points3D.bin  -> ...
#

set -euo pipefail

modes=(
    baseline
    unimodal
    gmm
    unimodal_all
    gmm_all
    unimodal_local
    gmm_local
)

sequences=(
    09c1414f1b
    0d2ee665be
    13c3e046d7
    1ada7a0617
    21d970d8de
    25f3b7a318
    27dd4da69e
    286b55a2bf
    31a2c91c43
)

amb3r_root=$SCRATCH/datasets/scannetpp/data/amb3r
sfm_root=$SCRATCH/experiments/depth-aware-ba/scannetpp_amb3r
out_root=$SCRATCH/experiments/depth-aware-ba/3dgs_data_snpp/dslr

n_built=0
problems=()

for seq in "${sequences[@]}"; do
    images=$amb3r_root/$seq/dslr/images
    if [[ ! -d $images ]]; then
        problems+=("$seq: no images dir at $images")
        continue
    fi
    n_img=$(( $(find "$images" -maxdepth 1 -name '*.png' | wc -l) ))
    n_seq_built=0

    for mode in "${modes[@]}"; do
        model=$sfm_root/$seq/dslr/sfm_$mode/0
        dst=$out_root/${seq}_${mode}

        missing=""
        for f in cameras images points3D; do
            [[ -f $model/$f.bin ]] || missing="$missing $f.bin"
        done
        if [[ -n $missing ]]; then
            problems+=("$seq/$mode: missing$missing in $model")
            continue
        fi

        # More than one sub-model means the reconstruction fragmented and only
        # sub-model 0 gets trained — the arm is then not comparable to the rest.
        n_sub=$(( $(find "$sfm_root/$seq/dslr/sfm_$mode" -mindepth 1 -maxdepth 1 -type d | wc -l) ))
        if (( n_sub > 1 )); then
            problems+=("$seq/$mode: $n_sub sub-models, only 0 linked (arm not comparable)")
        fi

        mkdir -p "$dst/sparse/0"
        ln -sfn "$images" "$dst/images"
        for f in cameras images points3D; do
            ln -sfn "$model/$f.bin" "$dst/sparse/0/$f.bin"
        done
        n_built=$((n_built + 1))
        n_seq_built=$((n_seq_built + 1))
    done
    echo "$seq: $n_img images, $n_seq_built/${#modes[@]} arms linked"
done

echo
echo "built $n_built / $(( ${#sequences[@]} * ${#modes[@]} )) source trees under $out_root"
if (( ${#problems[@]} > 0 )); then
    echo "${#problems[@]} PROBLEM(S):"
    for p in "${problems[@]}"; do echo "  $p"; done
    exit 1
fi
echo "no problems found"
