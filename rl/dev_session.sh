#!/bin/sh
tmux new-session -d 'dev_session'
tmux split-window -v
tmux split-window -h
tmux new-window 'mutt'
tmux -2 attach-session -d