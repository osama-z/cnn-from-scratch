"""Generate a standalone HTML gallery of real NumPy/C/C++ digit predictions."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np

from .verify import ROOT, parse_trace, verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "build/trained")
    args = parser.parse_args()
    verify_bundle(args.out)
    with np.load(args.out / "samples.npz", allow_pickle=False) as samples:
        images, labels = samples["images"], samples["labels"]
    with np.load(args.out / "reference.npz", allow_pickle=False) as reference:
        scores = {"NumPy": reference["softmax"].tolist()}
    for name, title in (("infer_c", "C"), ("infer_cpp", "C++")):
        raw = subprocess.check_output([str(ROOT / "trained" / name), str(args.out / "weights.bin"),
                                       str(args.out / "samples.bin")], text=True)
        trace = parse_trace(raw)
        scores[title] = [trace[(i, "softmax")].tolist() for i in range(len(images))]
    data = json.dumps({"images": np.rint(images[:, 0] * 255).astype(int).tolist(),
                       "labels": labels.tolist(), "scores": scores})
    html = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>CNN from scratch — digit demo</title>
<style>
body{font:17px system-ui,sans-serif;background:#101828;color:#f2f4f7;margin:0;padding:40px 24px}
main{max-width:800px;margin:auto}h1{font-size:36px;margin-bottom:8px}p{color:#cbd5e1;line-height:1.6}
.demo{display:flex;gap:32px;flex-wrap:wrap;margin-top:28px}canvas{width:280px;height:280px;image-rendering:pixelated;border:1px solid #475467;border-radius:12px}
table{border-collapse:collapse;min-width:280px}td,th{text-align:left;padding:14px;border-bottom:1px solid #344054}
input{width:100%;margin:24px 0}button{background:#a7f3d0;border:0;border-radius:8px;padding:10px 20px;font:inherit;cursor:pointer}
#truth{font-size:22px;color:#a7f3d0}small{color:#cbd5e1}
</style><main>
<p>NUMPY → C → C++</p><h1>One trained CNN. Three implementations.</h1>
<p>Explore held-out MNIST digits and compare predictions from the same exported weights.</p>
<div class="demo"><canvas id="digit" aria-label="Handwritten digit"></canvas><div>
<p id="truth" aria-live="polite"></p><table><thead><tr><th>Engine</th><th>Prediction</th><th>Probability</th></tr></thead><tbody id="scores"></tbody></table></div></div>
<label for="sample">Choose a test image <span id="position"></span></label>
<input type="range" id="sample" min="0" value="0"><button id="next">Next digit</button>
<p><small>Predictions were computed by the Python model and compiled C/C++ programs when this page was generated. This gallery works offline.</small></p>
</main><script>
const data=__DATA__;
const slider=document.getElementById('sample');slider.max=data.labels.length-1;
function show(){
const i=Number(slider.value),img=data.images[i],canvas=document.getElementById('digit');
canvas.width=img[0].length;canvas.height=img.length;const ctx=canvas.getContext('2d');
img.forEach((row,y)=>row.forEach((v,x)=>{ctx.fillStyle=`rgb(${v},${v},${v})`;ctx.fillRect(x,y,1,1)}));
document.getElementById('truth').textContent=`Actual digit: ${data.labels[i]}`;
document.getElementById('position').textContent=`${i+1} / ${data.labels.length}`;
document.getElementById('scores').replaceChildren();
for(const [name,values] of Object.entries(data.scores)){
const scores=values[i],p=Math.max(...scores),prediction=scores.indexOf(p),tr=document.createElement('tr');
for(const value of [name,String(prediction),`${(p*100).toFixed(1)}%`]){const td=document.createElement('td');td.textContent=value;tr.appendChild(td)}
document.getElementById('scores').appendChild(tr)}
}slider.addEventListener('input',show);document.getElementById('next').addEventListener('click',()=>{slider.value=(Number(slider.value)+1)%data.labels.length;show()});show();
</script></html>"""
    target = args.out / "demo.html"
    target.write_text(html.replace("__DATA__", data))
    print(f"Open {target}")


if __name__ == "__main__":
    main()
