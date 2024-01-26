# Title
*** sss ***

tmux new-session -s odfn_card_7

conda activate odfn
tmux attach -t odfn_card_7

tmux set-option -g mouse-select-pane on
tmux set-option -g mouse-resize-pane on
tmux set-option -g mouse-select-window o

python scripts/odfn/generating/generatefakeimages.py