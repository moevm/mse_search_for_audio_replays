#!/bin/sh

f=${1:-'-'}

gnuplot <<EOF
set xlabel 'time, sec'
set ylabel 'freq, Hz' offset 3
set yrange [0:8000]
set view 0, 0
set colorbox user origin 0.025, 0.3 size 0.04, 0.4
set zrange [0:1e6]
set cbrange [0:1e6]
set lmargin at screen 0.20
set tmargin at screen 0.85
set bmargin at screen 0.15
set rmargin at screen 0.80
splot '$f' nonuniform matrix with pm3d
pause mouse keypress
EOF
