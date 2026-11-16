"""Replot the recorded experiment in English, directly from source data."""
from pathlib import Path
import sys,json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Polygon,Arc
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/front180_wake'))
from run import field, scenes
OUT=ROOT/'presentation/figures_en';OUT.mkdir(exist_ok=True)
DATA=ROOT/'experiments/front180_wake'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})

def fish(ax):
    pts=np.array([[.04,0],[-.12,.09],[-.38,.055],[-.8,.012],[-1,.09],[-.94,0],[-1,-.09],[-.8,-.012],[-.38,-.055],[-.12,-.09]])
    ax.add_patch(Polygon(pts,facecolor='#ffdb52',edgecolor='#222',zorder=8))
    ax.annotate('Fish head at (0, 0)\nInitial heading: +x',xy=(0,0),xytext=(-.85,-.65),color='white',fontsize=11,arrowprops={'arrowstyle':'->','color':'white'},zorder=9)

def bg(ax,center,xlim,ylim):
    x=np.linspace(*xlim,220);y=np.linspace(*ylim,130);xx,yy=np.meshgrid(x,y)
    v=field(np.stack([xx,yy],axis=-1),1.,np.array(center));mask=(xx-center[0])**2+(yy-center[1])**2<.35**2
    im=ax.pcolormesh(xx,yy,np.ma.masked_where(mask,np.linalg.norm(v,axis=-1)),vmin=0,vmax=.8,cmap='viridis',shading='auto')
    ax.streamplot(x,y,np.ma.masked_where(mask,v[:,:,0]),np.ma.masked_where(mask,v[:,:,1]),color='#eeeeee',density=1.15,linewidth=.6,arrowsize=.7)
    ax.add_patch(Circle(center,.35,facecolor='#bb6344',edgecolor='white',lw=1.5,zorder=6))
    ax.set(xlim=xlim,ylim=ylim,aspect='equal',xlabel='World x (m)',ylabel='World y (m)')
    return im

def setup():
    fig,axes=plt.subplots(1,2,figsize=(15.5,6),layout='constrained',gridspec_kw={'width_ratios':[1.55,1]})
    im=bg(axes[0],[3,0],(-2,4.2),(-2,2));fish(axes[0]);axes[0].text(2.65,.65,'Target cylinder\nDiameter: 0.7 m',color='white',fontsize=11)
    axes[0].set_title('Fish faces +x; background current flows toward -x\nPrescribed analytical wake at t = 1 s',fontsize=14)
    fig.colorbar(im,ax=axes[0],shrink=.7,label='Local water speed (m/s)')
    goals(axes[1]);fig.savefig(OUT/'wake_fish_setup_en.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5.2),layout='constrained');im=bg(ax,[3,0],(-2,4.2),(-2,2));fish(ax)
    fig.colorbar(im,ax=ax,label='Local water speed (m/s)',shrink=.75);fig.savefig(OUT/'wake_field_en.png',dpi=200);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,7),layout='constrained');goals(ax);fig.savefig(OUT/'target_layout_en.png',dpi=200);plt.close(fig)

def goals(ax):
    ax.set_facecolor('#eaf3f5')
    for sc in scenes().values():ax.add_patch(Circle(sc['target'],.35,facecolor='#c25b3d',alpha=.2,edgecolor='#853e2c'))
    ax.add_patch(Arc((0,0),10,10,theta1=-90,theta2=90,ls='--',color='#555'))
    ax.scatter([0],[0],c='#ffdb52',edgecolors='black',s=90,zorder=9);ax.arrow(0,0,1,0,width=.025,color='#a77b00',length_includes_head=True)
    ax.annotate('Fish starts\nat (0, 0)',xy=(0,0),xytext=(-1,-1.3),arrowprops={'arrowstyle':'->'},fontsize=11)
    ax.arrow(4.7,4.6,-1,0,width=.03,color='#2672a3',length_includes_head=True);ax.text(1.8,5.,'Background current: 0.4 m/s',fontsize=10)
    ax.set(xlim=(-1.4,5.7),ylim=(-5.7,5.7),aspect='equal',xlabel='World x (m)',ylabel='World y (m)')
    ax.set_title('100 targets across the forward 180 degrees\n25 bearings x 4 distances; one cylinder per trial',fontsize=13)

def vorticity():
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    x=np.linspace(-2,4.2,220);y=np.linspace(-1.8,1.8,130);xx,yy=np.meshgrid(x,y);pts=np.stack([xx,yy],axis=-1);h=.003
    v=field(pts,1.,[3,0]);dx=(field(pts+[h,0],1.,[3,0])-field(pts-[h,0],1.,[3,0]))/(2*h);dy=(field(pts+[0,h],1.,[3,0])-field(pts-[0,h],1.,[3,0]))/(2*h)
    curl=np.ma.masked_where((xx-3)**2+yy**2<.38**2,dx[:,:,1]-dy[:,:,0])
    im=ax.pcolormesh(xx,yy,curl,cmap='RdBu_r',vmin=-20,vmax=20,shading='auto')
    ax.quiver(xx[::9,::12],yy[::9,::12],v[::9,::12,0],v[::9,::12,1],color='#444',scale=12,width=.0025)
    ax.add_patch(Circle((3,0),.35,color='#755b50'));ax.scatter([0],[0],color='#e0ab00',edgecolors='black',s=70,zorder=8)
    ax.set(aspect='equal',xlabel='World x (m)',ylabel='World y (m)',title='Alternating rotation in the prescribed wake (t = 1 s)',xlim=(-2,4.2),ylim=(-1.8,1.8))
    fig.colorbar(im,ax=ax,label='Vorticity (1/s)');fig.savefig(OUT/'vorticity_en.png',dpi=200);plt.close(fig)

def results():
    fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained');x=np.arange(3)
    for i,mode in enumerate(['uniform','wake']):
        vals=[json.loads((DATA/f'{mode}_A{c:.2f}_summary.json').read_text())['arrived'] for c in [.38,.42,.75]]
        pos=x+(i-.5)*.3;ax.bar(pos,vals,.3,label=['Uniform head current','Prescribed analytical wake'][i],color=['#498db2','#de8f42'][i])
        for xx,v in zip(pos,vals):ax.text(xx,v+2,f'{v}/100',ha='center',fontsize=12)
    ax.set(xticks=x,xticklabels=['0.38','0.42','0.75'],xlabel='Maximum commanded amplitude (rad)',ylabel='Arrival rate (%)',ylim=(0,115))
    ax.grid(axis='y',alpha=.2);ax.legend(loc='upper left');fig.savefig(OUT/'arrival_rates_en.png',dpi=200);plt.close(fig)
    allsc=scenes()
    for cap in [.38,.42,.75]:
        traces=np.load(DATA/f'wake_A{cap:.2f}_traces.npz');rows={r['case_id']:r for r in csv.DictReader((DATA/f'wake_A{cap:.2f}.csv').open())}
        for key in ['F01_01','F12_01','F23_01']:
            tr=traces[key];g=allsc[key]['target'];fig,ax=plt.subplots(figsize=(5.4,4.0),layout='constrained')
            xl=(min(-1.2,float(tr[:,1].min())-.3),max(g[0]+.8,float(tr[:,1].max())+.3));yl=(min(-1.,g[1]-.8,float(tr[:,2].min())-.3),max(1.,g[1]+.8,float(tr[:,2].max())+.3))
            bg(ax,g,xl,yl);ax.plot(tr[:,1],tr[:,2],color='#ffda52',lw=1.3,zorder=7)
            ax.scatter([0],[0],c='white',s=35,zorder=8);ax.scatter(tr[-1,1],tr[-1,2],c='#e53935',s=30,zorder=8)
            fig.savefig(OUT/f'trace_A{cap:.2f}_{key}.png',dpi=180);plt.close(fig)

def morphology():
    # Render the actual XML geometry, rather than an invented fish drawing.
    names=[3,4,5,6,8];images=[]
    for n in names:
        m=mujoco.MjModel.from_xml_path(str(ROOT/f'fish_morphologies/fish_N{n}.xml'));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        # Show only the fish geometry, with the pool and unrelated goal hidden.
        for g in range(m.ngeom):
            name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or ''
            m.geom_group[g]=0 if name=='head' or name.startswith('segment_') else 5
        renderer=mujoco.Renderer(m,height=240,width=640)
        camera=mujoco.MjvCamera();camera.lookat[:]=[-.5,0,.5];camera.distance=.9;camera.azimuth=90;camera.elevation=-85
        options=mujoco.MjvOption();options.sitegroup[:]=0;options.geomgroup[5]=0
        renderer.update_scene(d,camera,options);im=renderer.render().copy();renderer.close()
        mask=im.max(axis=2)>30;ys,xs=np.where(mask)
        im=im[max(0,ys.min()-8):min(im.shape[0],ys.max()+9),max(0,xs.min()-8):min(im.shape[1],xs.max()+9)]
        images.append(im)
    fig,axes=plt.subplots(5,1,figsize=(10,7),layout='constrained')
    for ax,n,im in zip(axes,names,images):ax.imshow(im);ax.axis('off');ax.set_title(f'N = {n} actuated joints'+('   (wake benchmark)' if n==4 else ''),fontsize=15)
    fig.savefig(OUT/'fish_morphologies_en.png',dpi=180);plt.close(fig)

def amplitude_stack():
    fig,axes=plt.subplots(3,1,figsize=(12,5.6),layout='constrained')
    key='F12_01';g=[3,0]
    for ax,cap in zip(axes,[.38,.42,.75]):
        tr=np.load(DATA/f'wake_A{cap:.2f}_traces.npz')[key]
        bg(ax,g,(-9,4),(-1.2,1.2));ax.plot(tr[:,1],tr[:,2],color='#ffda52',lw=1.4,zorder=7)
        ax.scatter([0],[0],color='white',s=32,zorder=8);ax.scatter(tr[-1,1],tr[-1,2],color='red',s=26,zorder=8)
        ax.set_title(f'A <= {cap:.2f} rad: '+('arrived in 28.82 s' if cap==.75 else 'timeout at 60 s'),fontsize=13)
    fig.savefig(OUT/'straight_amplitudes_en.png',dpi=200);plt.close(fig)

if __name__=='__main__':
    setup();vorticity();results();morphology();amplitude_stack();print(OUT)
