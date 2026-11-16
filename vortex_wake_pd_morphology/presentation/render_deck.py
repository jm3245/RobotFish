"""Render a Google Slides PDF export for local visual inspection."""
import argparse
from pathlib import Path
import fitz
from PIL import Image, ImageDraw
p=argparse.ArgumentParser();p.add_argument('pdf');p.add_argument('out');a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
doc=fitz.open(a.pdf)
thumbs=[]
for i,page in enumerate(doc):
    pix=page.get_pixmap(matrix=fitz.Matrix(1.25,1.25))
    pix.save(str(out/f'slide-{i+1:02d}.png'))
    im=Image.open(out/f'slide-{i+1:02d}.png').convert('RGB');im.thumbnail((480,270));thumbs.append(im)
sheet=Image.new('RGB',(960,((len(thumbs)+1)//2)*290),'#dddddd');draw=ImageDraw.Draw(sheet)
for i,im in enumerate(thumbs):
    x=i%2*480;y=i//2*290;sheet.paste(im,(x,y+20));draw.text((x+8,y+4),f'Slide {i+1}',fill='black')
sheet.save(out/'contact.png');print(f'Rendered {len(doc)} slides to {out}')
