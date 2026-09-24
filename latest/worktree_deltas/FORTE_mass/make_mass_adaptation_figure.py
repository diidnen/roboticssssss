#!/usr/bin/env python3
"""Create a compact mass-policy comparison figure after formal analysis."""
import argparse,csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--table",type=Path,required=True);ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    with a.table.open(newline="",encoding="utf-8") as f:rows=list(csv.DictReader(f))
    labels=[r["policy"] for r in rows];sr=[float(r["full_task_sr"]) for r in rows];force=[float(r["mean_force_N"]) for r in rows]
    fig,ax=plt.subplots(figsize=(9.4,4.6));x=np.arange(len(labels));w=.36
    ax.bar(x-w/2,sr,w,label="Full-task SR",color="#2f5d8c");ax.bar(x+w/2,np.asarray(force)/4,w,label="Mean force / 4 N",color="#d08b3e")
    ax.set_ylim(0,1.08);ax.set_ylabel("Rate / normalized force");ax.set_xticks(x,labels,rotation=18,ha="right");ax.grid(axis="y",color="#d9dee5",linewidth=.6);ax.spines[["top","right"]].set_visible(False);ax.legend(frameon=False,ncol=2,loc="upper left");fig.suptitle("Task2 mass force adaptation (paired heldout roots)",color="#222222");fig.subplots_adjust(bottom=.30,left=.08,right=.98,top=.86)
    a.out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,dpi=180,facecolor="white");plt.close(fig)
if __name__=="__main__":main()
