for task in drawer-open assembly button-press door-close door-open push reach
do
  for seed in 0 1 2
  do
    for alpha in 1.0
    do
      sbatch -p general --cpus-per-task=4 --mem=32G --gres=gpu:1 -d singleton -J ${task}_${seed}_${alpha} srun.sh ${task} ${seed} ${alpha}
    done
  done
done
