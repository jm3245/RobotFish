"""Field, setup, benchmark bars and real simulated head trajectories."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Arc
import numpy as np
from run import HERE, field, scenes

plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False

def fish(ax,p=(0,0),scale=1.,text_color='white'):
    outline=np.array([[.04,0],[-.12,.09],[-.38,.055],[-.8,.012],[-1.,.09],[-.94,0],[-1.,-.09],[-.8,-.012],[-.38,-.055],[-.12,-.09]])*scale+np.asarray(p)
    ax.add_patch(Polygon(outline,facecolor='#ffdb52',edgecolor='#222',zorder=8))
    ax.annotate('鱼头初始位置 (0,0) →',xy=p,xytext=(p[0]-.8,p[1]-.45),fontsize=9,
                arrowprops=dict(arrowstyle='->',color=text_color),color=text_color,zorder=9)

def background(ax,center,xlim,ylim,t=1.,colorbar=True):
    x=np.linspace(*xlim,190);y=np.linspace(*ylim,120);xx,yy=np.meshgrid(x,y)
    v=field(np.stack([xx,yy],axis=-1),t,np.asarray(center));speed=np.linalg.norm(v,axis=-1)
    mask=(xx-center[0])**2+(yy-center[1])**2<.35**2
    speed=np.ma.masked_where(mask,speed)
    im=ax.pcolormesh(xx,yy,speed,cmap='viridis',vmin=0,vmax=.8,shading='auto')
    ax.streamplot(x,y,np.ma.masked_where(mask,v[:,:,0]),np.ma.masked_where(mask,v[:,:,1]),color='#eeeeee',density=1.2,linewidth=.55,arrowsize=.7)
    ax.add_patch(Circle(center,.35,facecolor='#bd6344',edgecolor='white',lw=1.3,zorder=6))
    ax.set(xlim=xlim,ylim=ylim,aspect='equal',xlabel='世界 x (m)',ylabel='世界 y (m)')
    return im

def preview():
    fig,axes=plt.subplots(1,2,figsize=(13.5,5),layout='constrained')
    im=background(axes[0],[3,0],(-2,4.2),(-2,2))
    fish(axes[0]);axes[0].text(2.7,.6,'目标圆柱\n直径 0.7 m',color='white',fontsize=10)
    axes[0].set_title('正前方目标：鱼朝 +x，背景流朝 −x\n解析漩涡场 t=1 s（已用于鱼体受力）')
    fig.colorbar(im,ax=axes[0],shrink=.65,label='局部流速 (m/s)')
    ax=axes[1];ax.set_facecolor('#eaf3f5')
    for sc in scenes().values():
        ax.add_patch(Circle(sc['target'],.35,facecolor='#c25b3d',alpha=.18,edgecolor='#853e2c'))
    ax.add_patch(Arc((0,0),10,10,theta1=-90,theta2=90,linestyle='--',color='#555'))
    ax.arrow(0,0,1,0,width=.025,color='#b38200',length_includes_head=True,zorder=8)
    ax.scatter([0],[0],c='#ffdb52',edgecolors='black',s=90,zorder=9)
    ax.annotate('鱼初始朝向 +x',xy=(0,0),xytext=(-1,-1),arrowprops=dict(arrowstyle='->'))
    ax.arrow(3.9,4.5,-1,0,width=.03,color='#2672a3',length_includes_head=True)
    ax.text(2.4,4.85,'统一背景流 −x：0.4 m/s',fontsize=9)
    ax.set(xlim=(-1.3,5.6),ylim=(-5.6,5.6),aspect='equal',xlabel='世界 x (m)',ylabel='世界 y (m)',
           title='100 个独立目标：前方 180° 的 25 个角度 × 4 个距离\n每场仅放置其中一个圆柱，图中叠加展示全部位置')
    fig.savefig(HERE/'wake_fish_setup.png',dpi=170);plt.close(fig)
    # Signed vorticity makes alternating rotations visible even when the
    # background current prevents closed streamlines in the world frame.
    fig,ax=plt.subplots(figsize=(10,4.4),layout='constrained')
    x=np.linspace(-2,4.2,220);y=np.linspace(-1.8,1.8,130);xx,yy=np.meshgrid(x,y)
    pts=np.stack([xx,yy],axis=-1);v=field(pts,1.,np.array([3.,0.]))
    h=.003
    dx=(field(pts+[h,0],1.,[3,0])-field(pts-[h,0],1.,[3,0]))/(2*h)
    dy=(field(pts+[0,h],1.,[3,0])-field(pts-[0,h],1.,[3,0]))/(2*h)
    curl=dx[:,:,1]-dy[:,:,0]
    mask=(xx-3)**2+yy**2<.38**2
    im=ax.pcolormesh(xx,yy,np.ma.masked_where(mask,curl),cmap='RdBu_r',vmin=-20,vmax=20,shading='auto')
    ax.quiver(xx[::9,::12],yy[::9,::12],v[::9,::12,0],v[::9,::12,1],color='#4b4b4b',scale=12,width=.0025)
    ax.add_patch(Circle((3,0),.35,facecolor='#755b50',edgecolor='black',zorder=6));fish(ax,text_color='black')
    ax.set(xlim=(-2,4.2),ylim=(-1.8,1.8),aspect='equal',xlabel='世界 x (m)',ylabel='世界 y (m)',title='涡量：红色逆时针、蓝色顺时针；箭头是局部水速（t=1 s）')
    fig.colorbar(im,ax=ax,label='涡量 ∂v/∂x − ∂u/∂y (1/s)')
    fig.savefig(HERE/'wake_vorticity_fish.png',dpi=180);plt.close(fig)

def results():
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');x=np.arange(3)
    for i,mode in enumerate(['uniform','wake']):
        values=[]
        for cap in [.38,.42,.75]:
            data=json.loads((HERE/f'{mode}_A{cap:.2f}_summary.json').read_text())
            assert data['n']==100
            values.append(data['arrived'])
        pos=x+(i-.5)*.3
        ax.bar(pos,values,.3,label='均匀逆流' if mode=='uniform' else '解析圆柱漩涡',color=['#498db2','#de8f42'][i])
        for xx,v in zip(pos,values):ax.text(xx,v+1,f'{v}%',ha='center')
    ax.set(xticks=x,xticklabels=['0.38','0.42','0.75'],xlabel='最大命令摆幅 (rad)，频率固定 2.5 Hz',ylabel='到达圆柱表面的比例',ylim=(0,110),title='最新版 6D PD：前方半圆 100 场景，背景流 0.4 m/s')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.savefig(HERE/'arrival_rates.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
    allsc=scenes()
    for i,cap in enumerate([.38,.42,.75]):
        traces=np.load(HERE/f'wake_A{cap:.2f}_traces.npz')
        rows={r['case_id']:r for r in csv.DictReader((HERE/f'wake_A{cap:.2f}.csv').open(encoding='utf-8'))}
        for j,key in enumerate(['F01_01','F12_01','F23_01']):
            ax=axes[i,j];tr=traces[key];sc=allsc[key];g=sc['target']
            # Show the full actual trajectory; flow image is a t=1 s snapshot.
            xl=(min(-1.2,float(tr[:,1].min())-.3),max(g[0]+.8,float(tr[:,1].max())+.3))
            yl=(min(-1.,g[1]-.8,float(tr[:,2].min())-.3),max(1.,g[1]+.8,float(tr[:,2].max())+.3))
            background(ax,g,xl,yl)
            ax.plot(tr[:,1],tr[:,2],color='#ffda52',lw=1.3,zorder=7)
            ax.scatter([0],[0],c='white',s=25,zorder=8);ax.scatter(tr[-1,1],tr[-1,2],c='red',s=22,zorder=8)
            ax.set_title(f"A≤{cap} rad；目标角 {sc['angle_deg']:.1f}°\n{'到达' if int(rows[key]['success']) else '未到达'}，t={rows[key]['seconds']} s",fontsize=10)
    fig.suptitle('实际 MuJoCo 鱼头轨迹（黄线）；底图为 t=1 s 流场快照；红点为末次记录位置')
    fig.savefig(HERE/'wake_trajectories_9.png',dpi=170);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preview-only',action='store_true');a=p.parse_args()
    preview()
    if not a.preview_only:results()
