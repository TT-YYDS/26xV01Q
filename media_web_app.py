import os, sys, uuid, shutil, json, hashlib, re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
if (ROOT / '.deps').exists():
    sys.path.insert(0, str(ROOT / '.deps'))

from flask import Flask, request, render_template_string, redirect, url_for, flash, send_file, Response, jsonify
from werkzeug.utils import secure_filename
import ezdxf
import open3d as o3d
from media_analysis_core import (
    analyze,
    fit_profile_global,
    save_height_colored_cad_overlay_image,
    sample_entity,
    section_profile_from_points,
)

app = Flask(__name__)
app.secret_key = os.environ.get("POINTCLOUD_WEB_SECRET_KEY")
if app.secret_key is None or len(app.secret_key.strip()) < 32:
    raise RuntimeError(
        "Set POINTCLOUD_WEB_SECRET_KEY to a private random value of at least 32 characters before starting the web app."
    )
WORK = ROOT / 'web_runs'
WORK.mkdir(exist_ok=True)
POINTCLOUD_PREVIEW_ROOT = WORK / 'pointcloud_previews'
POINTCLOUD_PREVIEW_ROOT.mkdir(exist_ok=True)
EXTERNAL_SECTION_MODE = 'unified_final_section'

HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Media偏差分析</title>
<style>
body{font-family:Arial,"Microsoft YaHei";background:#f3f6f8;color:#17202a;margin:0}
.wrap{max-width:1480px;margin:24px auto;padding:0 16px}
.card{background:white;padding:20px;margin:15px 0;border-radius:12px;box-shadow:0 2px 10px #0001}
label{display:block;margin:9px 0 4px;font-weight:bold}
input,button{padding:9px;font-size:15px}
button{background:#087f5b;color:#fff;border:0;border-radius:5px;margin:8px 5px 8px 0;cursor:pointer}
button:disabled{background:#9aa8b2;cursor:wait}
.muted{color:#667}
.hint{padding:9px;background:#fff8df;border-radius:6px;margin:8px 0}
.hint.error{color:#a11;background:#fee}
.upload-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;margin-top:12px}
.upload-field{min-width:0;padding:13px 14px;border:1px solid #d5dde5;border-radius:10px;background:#fbfcfd}
.upload-field label{margin:0 0 9px}
.upload-field input[type=file]{box-sizing:border-box;width:100%;max-width:100%;padding:8px;background:#fff;border:1px solid #cbd5dc;border-radius:6px}
.upload-field small{display:block;min-height:18px;margin-top:8px}
.compare-status-row{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;margin-top:14px}
.compare-status-row .hint{margin:0;min-height:42px;box-sizing:border-box}
.compare-controls{margin-top:12px;padding:12px 14px;border:1px solid #d5dde5;border-radius:10px;background:#fbfcfd}
.compare-controls .tool{margin-bottom:4px}
.compare-controls .hint{margin:5px 0 0}
.reference-grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;align-items:stretch;margin-top:14px}
.reference-panel{display:grid;grid-template-rows:56px minmax(0,1fr) 62px;border:1px solid #d5dde5;border-radius:10px;padding:0;background:#fbfcfd;min-width:0;overflow:hidden}
.viewer-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:0 15px;border-bottom:1px solid #d5dde5;background:#f7fafc}
.reference-panel h3{margin:0;font-size:17px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.viewer-badge{flex:0 0 auto;padding:3px 8px;border-radius:999px;background:#e7f5ee;color:#087f5b;font-size:12px;font-weight:bold}
.viewer-canvas,.cadbox{position:relative;height:620px;min-height:0;overflow:hidden;background:#fff}
.preview-placeholder{box-sizing:border-box;width:100%;height:100%;min-height:0;display:flex;align-items:center;justify-content:center;text-align:center;padding:20px;border:0;background:#fff;color:#667}
.preview-placeholder.hidden{display:none}
.pointcloud-preview{display:block;width:100%;height:100%;max-height:none;object-fit:contain;border:0;background:#fff}
.pointcloud-preview.hidden{display:none}
.viewer-footer{box-sizing:border-box;height:62px;min-height:62px;padding:8px 12px;border-top:1px solid #d5dde5;color:#53616d;overflow:auto}
.cadbox{border:0}
.cadbox svg{width:100%;height:100%;display:block}
.tool button.active{background:#d94841}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.metric{background:#edf8f3;padding:12px}
.metric b{display:block;font-size:21px}
.curve{width:100%;display:block}
.table{width:100%;border-collapse:collapse}
.table td,.table th{padding:8px;border-bottom:1px solid #ddd;text-align:left}
.err{color:#a11;background:#fee;padding:10px;border-radius:5px}
.region-item{margin:3px 0}
.small{font-size:13px}
@media(max-width:900px){.upload-grid,.compare-status-row,.reference-grid{grid-template-columns:1fr}.viewer-canvas,.cadbox{height:520px}.metrics{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<div class="wrap">
<h1>Media 点云 / CAD 绿色边偏差分析</h1>
<p class="muted">左侧始终保留最终点云原始坐标和排列；整张 CAD 轮廓向固定点云拟合，蓝色线段只作为辅助锚定，不会旋转或重排点云。</p>
{% for m in get_flashed_messages() %}<div class="err">{{m}}</div>{% endfor %}
<div class="card">
<form id="analysisForm" method="post" enctype="multipart/form-data">
<div class="upload-grid">
<section class="upload-field">
<label for="pointcloudInput">点云文件（可多选 PLY/PCD/XYZ/CSV）</label>
<input id="pointcloudInput" type="file" name="pointcloud" multiple>
<small id="pointcloudFiles" class="muted">可上传 1 个已处理截面或 5 个原始点云；也可直接接收本项目拼接流程的最终侧面截面。</small>
</section>
<section class="upload-field">
<label for="dxfInput">CAD 图纸（DXF）</label>
<input id="dxfInput" type="file" name="dxf" accept=".dxf" required>
<small class="muted">载入后将在右侧显示可点击选择的 CAD 图纸。</small>
</section>
</div>
<input type="hidden" name="pointcloud_preview_token" id="pointcloudPreviewToken">
<div class="compare-status-row">
<div id="pointcloudPreviewStatus" class="hint">选择点云后，系统会先完成去噪、快速配准、GPU/CPU 自动融合和垂直截面处理。</div>
<div id="cadStatus" class="hint">请先选择 DXF，等待图纸预览加载。</div>
</div>
<div class="compare-controls">
<div class="tool"><button type="button" data-mode="x">选择绿色 X 边</button><button type="button" data-mode="y">选择绿色 Y 边</button><button type="button" data-mode="fit">选择蓝色拟合轮廓（起止）</button><button type="button" id="finishFit">完成蓝色拟合选择</button><button type="button" id="clearRegions">清除选择</button></div>
<div class="hint">左图点云保持不变，CAD 仅拟合到点云坐标。选择蓝色拟合轮廓时，先点击起始线段，再点击终点线段；两者端点连续相连的 CAD 图元会自动选中。每完成一组起止线段，可直接继续选择下一组并追加。</div>
</div>
<div class="reference-grid">
<section class="reference-panel">
<div class="viewer-heading"><h3 id="pointcloudPreviewTitle">固定点云 / CAD 拟合图</h3><span class="viewer-badge">点云固定</span></div>
<div class="viewer-canvas"><div id="pointcloudPreviewPlaceholder" class="preview-placeholder">这里会显示处理后的固定点云<br>加载 DXF 后仅将 CAD 拟合到点云</div><img id="pointcloudPreviewImage" class="pointcloud-preview hidden" alt="固定点云与 CAD 拟合图"></div>
<div id="pointcloudPreviewMeta" class="viewer-footer small"></div>
</section>
<section class="reference-panel">
<div class="viewer-heading"><h3>CAD 图纸（点击选择线段）</h3><span class="viewer-badge">拟合到点云</span></div>
<div id="cadbox" class="cadbox"></div>
<div id="regionList" class="viewer-footer small">尚未选择 CAD 线段。</div>
</section>
</div>
<input type="hidden" name="selected_entities" id="selectedEntities">
<label>CAD 与点云拟合采样步长</label>
<input type="number" name="step" value=".05" min=".005" step=".005">
<br><button id="submitButton" type="submit">开始分析</button>
</form>
</div>
{% if report %}
<div class="card">
<h2>结果</h2>
{% if report.outputs and report.outputs.cad_alignment_image %}<h3>固定最终点云 / CAD 拟合对照图</h3><div class="hint">点云保持最终截面原始坐标；整张 CAD 轮廓拟合到固定点云，蓝色线段仅作为辅助锚定。 CAD&#8594;点云旋转：<b>{{'%+.4f'|format(report.fit.cad_to_pointcloud_angle_deg)}}&#176;</b>；CAD 左右镜像：<b>{% if report.fit.mirror_applied %}是{% else %}否{% endif %}</b>。</div><img class="curve" src="{{url_for('artifact',run=run_id,name='pointcloud_cad_alignment.png')}}" alt="固定点云与 CAD 拟合对照图"><p class="small muted">蓝色点云为未变换的最终截面；灰色为其他 CAD 线段，蓝色为拟合轮廓，绿色为测量边。</p>{% endif %}
<h3>综合图片报告</h3>
<img class="curve" src="{{url_for('artifact',run=run_id,name='media_measurement_report.png')}}" alt="综合图片报告">
<div class="metrics">{% for key in ['x_axis_edge','y_axis_edge'] %}{% set r=report.green_measurements[key] %}<div class="metric">{{r.label}} 最大<b>{{'%.4f'|format(r.max_signed)}}</b></div><div class="metric">{{r.label}} 最小<b>{{'%.4f'|format(r.min_signed)}}</b></div>{% endfor %}</div>
<table class="table"><tr><th>边</th><th>全长</th><th>平均</th><th>平均绝对</th><th>最大绝对</th><th>最小/最大位置</th><th>角度差</th><th>覆盖率</th></tr>{% for key in ['x_axis_edge','y_axis_edge'] %}{% set r=report.green_measurements[key] %}<tr><td>{{r.label}}</td><td>{{'%.4f'|format(r.length)}}</td><td>{{'%+.4f'|format(r.mean_signed)}}</td><td>{{'%.4f'|format(r.mean_absolute)}}</td><td>{{'%.4f'|format(r.max_absolute)}}</td><td>{{'%.4f'|format(r.min_station)}} / {{'%.4f'|format(r.max_station)}}</td><td>{{'%+.4f°'|format(r.signed_angle_difference_deg)}}</td><td>{{'%.1f%%'|format(r.coverage_ratio*100)}}</td></tr>{% endfor %}</table>
</div>
{% endif %}
</div>
<script>
const initialExternalPreview={{ external_preview|tojson }};
const pointcloudInput=document.getElementById('pointcloudInput');
const pointcloudFiles=document.getElementById('pointcloudFiles');
const pointcloudPreviewToken=document.getElementById('pointcloudPreviewToken');
const pointcloudPreviewTitle=document.getElementById('pointcloudPreviewTitle');
const pointcloudPreviewStatus=document.getElementById('pointcloudPreviewStatus');
const pointcloudPreviewPlaceholder=document.getElementById('pointcloudPreviewPlaceholder');
const pointcloudPreviewImage=document.getElementById('pointcloudPreviewImage');
const pointcloudPreviewMeta=document.getElementById('pointcloudPreviewMeta');
const submitButton=document.getElementById('submitButton');
const input=document.getElementById('dxfInput');
const box=document.getElementById('cadbox');
const status=document.getElementById('cadStatus');
const hidden=document.getElementById('selectedEntities');
const list=document.getElementById('regionList');
const form=document.getElementById('analysisForm');
const modeButtons=[...document.querySelectorAll('[data-mode]')];
const finishButton=document.getElementById('finishFit');
let svg=null,mode=null,pointcloudPreviewBusy=false,cadDirectionPreviewBusy=false,previewRequestId=0,cadPreviewRequestId=0,cadDirectionRequestId=0;
let fitTopology=null;
let selection={x:[],y:[],fit:[]};
let fitRangeStart=null;

hidden.name='selected_entities';
const selectionStyle=document.createElement('style');
selectionStyle.textContent='.cad-entity{cursor:pointer;transition:stroke .12s ease,stroke-width .12s ease}.cad-entity.selected-x{stroke:#00a651!important;stroke-width:4px!important}.cad-entity.selected-y{stroke:#7acb00!important;stroke-width:4px!important}.cad-entity.selected-fit{stroke:#075cff!important;stroke-width:3px!important}.cad-entity.fit-range-start{stroke:#f59e0b!important;stroke-width:4px!important;stroke-dasharray:8 4}.cad-entity:focus{outline:none}';
document.head.appendChild(selectionStyle);

function setPreviewStatus(message,error=false){
  pointcloudPreviewStatus.textContent=message;
  pointcloudPreviewStatus.classList.toggle('error',error);
}
function updateSubmitAvailability(){
  submitButton.disabled=pointcloudPreviewBusy||cadDirectionPreviewBusy;
}
function clearPointcloudPreview(message){
  pointcloudPreviewToken.value='';
  pointcloudPreviewTitle.textContent='融合后点云侧面参考图';
  pointcloudPreviewImage.removeAttribute('src');
  pointcloudPreviewImage.classList.add('hidden');
  pointcloudPreviewPlaceholder.classList.remove('hidden');
  pointcloudPreviewPlaceholder.innerHTML=message||'这里会显示处理后的点云侧面轮廓<br>请根据这张图判断 CAD 线段位置';
  pointcloudPreviewMeta.textContent='';
}
function setPreviewBusy(busy){
  pointcloudPreviewBusy=busy;
  updateSubmitAvailability();
}
function setCadDirectionPreviewBusy(busy){
  cadDirectionPreviewBusy=busy;
  updateSubmitAvailability();
}
async function refreshCadDirectionPreview(preferSelectedFit=false){
  const dxfFile=input.files[0];
  const token=pointcloudPreviewToken.value;
  if(!dxfFile||!token||pointcloudPreviewBusy)return;
  const fitEntities=preferSelectedFit&&selection.fit.length?[...selection.fit]:[];
  const requestId=++cadDirectionRequestId;
  const upload=new FormData();
  upload.append('pointcloud_preview_token',token);
  upload.append('dxf',dxfFile,dxfFile.name);
  if(fitEntities.length)upload.append('fit_entities',JSON.stringify(fitEntities));
  setCadDirectionPreviewBusy(true);
  setPreviewStatus(fitEntities.length?'\u6b63\u5728\u5c06\u6574\u5f20 CAD \u8f6e\u5ed3\u62df\u5408\u5230\u56fa\u5b9a\u70b9\u4e91\uff0c\u84dd\u8272\u7ebf\u6bb5\u4ec5\u4f5c\u4e3a\u8f85\u52a9\u951a\u5b9a\u2026\u2026':'\u6b63\u5728\u5c06 CAD \u6574\u4f53\u8f6e\u5ed3\u62df\u5408\u5230\u56fa\u5b9a\u70b9\u4e91\u2026\u2026');
  try{
    const response=await fetch('/pointcloud_cad_direction_preview',{method:'POST',body:upload});
    const data=await response.json().catch(()=>({error:'服务器返回内容无法解析'}));
    if(requestId!==cadDirectionRequestId)return;
    if(!response.ok||!data.image_url){
      setPreviewStatus('CAD \u5bf9\u7167\u56fe\u751f\u6210\u5931\u8d25\uff1a'+(data.error||'\u672a\u77e5\u9519\u8bef')+'\uff1b\u4ecd\u53ef\u4f7f\u7528\u539f\u59cb\u70b9\u4e91\u4fa7\u9762\u56fe\uff0c\u6b63\u5f0f\u5206\u6790\u5c06\u4f7f\u7528\u5168\u56fe CAD \u8f6e\u5ed3\u5bf9\u9f50\u3002',true);
      return;
    }
    const anchored=data.alignment_source==='fixed_pointcloud_with_selected_blue_anchor';
    pointcloudPreviewTitle.textContent=anchored?'\u56fa\u5b9a\u70b9\u4e91 / CAD \u62df\u5408\u56fe\uff08\u84dd\u7ebf\u951a\u5b9a\uff09':'\u56fa\u5b9a\u70b9\u4e91 / CAD \u62df\u5408\u56fe';
    pointcloudPreviewImage.src=data.image_url+'?t='+Date.now();
    pointcloudPreviewImage.classList.remove('hidden');
    pointcloudPreviewPlaceholder.classList.add('hidden');
    const mirrorText=data.mirror_applied?'；已左右镜像':'';
    const sourceText=anchored?'\u5df2\u5c06\u6574\u5f20 CAD \u8f6e\u5ed3\u62df\u5408\u5230\u56fa\u5b9a\u70b9\u4e91\uff0c\u4f7f\u7528 '+data.selected_fit_count+' \u6761\u84dd\u8272\u7ebf\u6bb5\u8f85\u52a9\u951a\u5b9a':'\u5df2\u5c06\u6574\u5f20 CAD \u8f6e\u5ed3\u62df\u5408\u5230\u56fa\u5b9a\u70b9\u4e91';
    setPreviewStatus(sourceText+'。点云坐标和排列保持不变'+mirrorText+'。');
    const heightText=data.height_low!==undefined&&data.height_high!==undefined?'；高度彩色：Z '+Number(data.height_low).toFixed(3)+'（低）→ '+Number(data.height_high).toFixed(3)+'（高）':'';
    pointcloudPreviewMeta.textContent=sourceText+'；CAD→点云旋转：'+Number(data.cad_to_pointcloud_angle_deg).toFixed(4)+'°'+mirrorText+heightText+'。';
  }catch(error){
    if(requestId===cadDirectionRequestId)setPreviewStatus('CAD 对照图生成失败：'+error.message+'；仍可使用原始点云侧面图。',true);
  }finally{
    if(requestId===cadDirectionRequestId)setCadDirectionPreviewBusy(false);
  }
}
pointcloudInput.addEventListener('change',async()=>{
  const requestId=++previewRequestId;
  ++cadDirectionRequestId;
  setCadDirectionPreviewBusy(false);
  const files=[...pointcloudInput.files];
  setPreviewBusy(false);
  clearPointcloudPreview();
  if(!files.length){
    pointcloudFiles.textContent='按住 Ctrl 或 Shift 选择多个点云；原始 Media 点云请一次选择 5 个文件。';
    setPreviewStatus('请选择点云文件。');
    return;
  }
  pointcloudFiles.textContent='已选择 '+files.length+' 个文件：'+files.map(file=>file.name).join('、');
  if(files.length!==1&&files.length!==5){
    setPreviewStatus('当前选择了 '+files.length+' 个文件；请上传 1 个已处理截面 PLY，或一次选择 5 个原始 Media 点云。',true);
    return;
  }
  const upload=new FormData();
  files.forEach(file=>upload.append('pointcloud',file,file.name));
  setPreviewBusy(true);
  setPreviewStatus(files.length===5?'正在快速配准 5 个点云；CUDA 可用时会自动使用 RTX GPU 融合并生成垂直截面，请等待……':'正在生成点云侧面参考图，请等待……');
  try{
    const response=await fetch('/pointcloud_preview',{method:'POST',body:upload});
    const data=await response.json().catch(()=>({error:'服务器返回内容无法解析'}));
    if(requestId!==previewRequestId)return;
    if(!response.ok||!data.image_url){
      setPreviewStatus('点云侧面预览失败：'+(data.error||'未知错误'),true);
      return;
    }
    pointcloudPreviewToken.value=data.token||'';
    pointcloudPreviewTitle.textContent='融合后点云侧面参考图';
    pointcloudPreviewImage.src=data.image_url+'?t='+Date.now();
    pointcloudPreviewImage.classList.remove('hidden');
    pointcloudPreviewPlaceholder.classList.add('hidden');
    const processing=data.processing||null;
    const totalSeconds=Number(processing&&processing.total_seconds);
    const durationText=Number.isFinite(totalSeconds)?'；总耗时 '+totalSeconds.toFixed(1)+' 秒':'';
    const gpuText=processing&&processing.gpu_used?'；已使用 '+(processing.gpu_device||'CUDA GPU')+' 加速融合':(processing&&processing.fusion_backend?'；融合后端：'+processing.fusion_backend:'');
    setPreviewStatus('融合/处理完成'+durationText+gpuText+'。载入 CAD 后只拟合 CAD，左侧点云不旋转、不重排。');
    pointcloudPreviewMeta.textContent=(data.mode||'点云处理')+'；文件数：'+data.file_count+'；截面点数：'+(data.section_points||'未统计')+durationText+gpuText;
  }catch(error){
    if(requestId===previewRequestId)setPreviewStatus('点云侧面预览失败：'+error.message,true);
  }finally{
    if(requestId===previewRequestId){
      setPreviewBusy(false);
      refreshCadDirectionPreview();
    }
  }
});

function emptySelection(){return {x:[],y:[],fit:[]};}
function setActiveButton(active){modeButtons.forEach(item=>item.classList.toggle('active',item===active));}
function setMode(nextMode,button){
  if(nextMode!=='fit')fitRangeStart=null;
  mode=nextMode;
  setActiveButton(button);
  drawSelections();
  if(mode==='fit')status.textContent='蓝色拟合轮廓：先点击起始线段，再点击终点线段；每完成一组后可继续选择下一组追加';
  else status.textContent=(mode==='x'?'绿色 X 边':'绿色 Y 边')+'：请点击一条 CAD 线段；再次点击可替换';
}
input.addEventListener('change',async()=>{
  if(!input.files[0])return;
  const requestId=++cadPreviewRequestId;
  ++cadDirectionRequestId;
  setCadDirectionPreviewBusy(false);
  const upload=new FormData(); upload.append('dxf',input.files[0]);
  status.textContent='正在读取 CAD 图纸……';
  try{
    const response=await fetch('/cad_preview',{method:'POST',body:upload});
    const data=await response.json().catch(()=>({error:'服务器返回内容无法解析'}));
    if(requestId!==cadPreviewRequestId)return;
    if(!response.ok||!data.svg){status.textContent='DXF 预览失败：'+(data.error||'未知错误');return;}
    box.innerHTML=data.svg; svg=box.querySelector('svg'); fitTopology=null; selection=emptySelection(); fitRangeStart=null; mode=null; setActiveButton(null);
    status.textContent='图纸已加载：请选择工具后直接点击 CAD 线段';
    bindCanvas(); drawSelections(); update();
    refreshCadDirectionPreview();
  }catch(error){
    if(requestId===cadPreviewRequestId)status.textContent='DXF 预览失败：'+error.message;
  }
});
modeButtons.forEach(button=>button.addEventListener('click',()=>setMode(button.dataset.mode,button)));
finishButton.addEventListener('click',()=>{
  if(fitRangeStart!==null){status.textContent='请再点击一条终点线段，完成蓝色拟合轮廓范围选择';return;}
  if(!selection.fit.length){status.textContent='还没有选择蓝色拟合线段';return;}
  mode=null; setActiveButton(null); status.textContent='蓝色拟合轮廓已保存：'+selection.fit.length+'条线段，正在按整张 CAD 轮廓全局对齐左图'; update();
  refreshCadDirectionPreview(true);
});
document.getElementById('clearRegions').addEventListener('click',()=>{
  selection=emptySelection(); fitRangeStart=null; mode=null; setActiveButton(null); drawSelections(); update(); status.textContent='选择已清除，左图已恢复为全图预对齐';
  refreshCadDirectionPreview();
});
function bindCanvas(){
  if(!svg)return;
  svg.querySelectorAll('.cad-entity').forEach(entity=>{
    entity.setAttribute('tabindex','0');
    entity.addEventListener('click',event=>selectEntity(event,entity));
    entity.addEventListener('keydown',event=>{
      if(event.key==='Enter'||event.key===' '){event.preventDefault();selectEntity(event,entity);}
    });
  });
}
function removeFromOtherSelections(index,keep){
  for(const key of Object.keys(selection))if(key!==keep)selection[key]=selection[key].filter(value=>value!==index);
}
function fitEntityGeometry(entity){
  const index=Number(entity.dataset.index);
  const start=[Number(entity.dataset.startX),Number(entity.dataset.startY)];
  const end=[Number(entity.dataset.endX),Number(entity.dataset.endY)];
  const length=Number(entity.dataset.length);
  if(!Number.isInteger(index)||!start.every(Number.isFinite)||!end.every(Number.isFinite))return null;
  return {index,endpoints:[start,end],length:Number.isFinite(length)&&length>0?length:1e-9};
}
function buildFitTopology(){
  if(!svg)return null;
  if(fitTopology&&fitTopology.svg===svg)return fitTopology;
  const tolerance=Math.max(Number(svg.dataset.connectTolerance)||0,1e-9);
  const toleranceSquared=tolerance*tolerance;
  const entities=[...svg.querySelectorAll('.cad-entity')].map(fitEntityGeometry).filter(Boolean);
  const byIndex=new Map(entities.map(entity=>[entity.index,entity]));
  const adjacency=new Map(entities.map(entity=>[entity.index,[]]));
  for(let first=0;first<entities.length;first+=1){
    const left=entities[first];
    for(let second=first+1;second<entities.length;second+=1){
      const right=entities[second];
      let connected=false;
      for(const leftPoint of left.endpoints){
        for(const rightPoint of right.endpoints){
          const dx=leftPoint[0]-rightPoint[0];
          const dy=leftPoint[1]-rightPoint[1];
          if(dx*dx+dy*dy<=toleranceSquared){connected=true;break;}
        }
        if(connected)break;
      }
      if(!connected)continue;
      const cost=(left.length+right.length)/2;
      adjacency.get(left.index).push({index:right.index,cost});
      adjacency.get(right.index).push({index:left.index,cost});
    }
  }
  fitTopology={svg,indices:entities.map(entity=>entity.index),byIndex,adjacency};
  return fitTopology;
}
function fitContourPathIndices(startIndex,endIndex){
  const topology=buildFitTopology();
  if(!topology||!topology.byIndex.has(startIndex)||!topology.byIndex.has(endIndex))return [];
  if(startIndex===endIndex)return [startIndex];
  const distances=new Map(topology.indices.map(index=>[index,Infinity]));
  const previous=new Map();
  const remaining=new Set(topology.indices);
  distances.set(startIndex,0);
  while(remaining.size){
    let current=null;
    let currentDistance=Infinity;
    for(const index of remaining){
      const distance=distances.get(index);
      if(distance<currentDistance){current=index;currentDistance=distance;}
    }
    if(current===null||!Number.isFinite(currentDistance))break;
    remaining.delete(current);
    if(current===endIndex)break;
    for(const edge of topology.adjacency.get(current)){
      if(!remaining.has(edge.index))continue;
      const candidate=currentDistance+edge.cost;
      if(candidate<distances.get(edge.index)){
        distances.set(edge.index,candidate);
        previous.set(edge.index,current);
      }
    }
  }
  if(!previous.has(endIndex))return [];
  const path=[endIndex];
  while(path[path.length-1]!==startIndex){
    const prior=previous.get(path[path.length-1]);
    if(prior===undefined)return [];
    path.push(prior);
  }
  return path.reverse();
}
function selectEntity(event,entity){
  if(!mode){status.textContent='请先选择绿色 X 边、绿色 Y 边或蓝色拟合轮廓工具';return;}
  const index=Number(entity.dataset.index);
  if(!Number.isInteger(index))return;
  if(mode==='fit'){
    if(fitRangeStart===null){
      fitRangeStart=index;
      status.textContent='已选择起始线段 '+index+'；请点击终点线段，端点连续相连的轮廓将自动选中';
    }else if(fitRangeStart===index){
      fitRangeStart=null;
      status.textContent='已取消蓝色拟合轮廓起始线段；请重新点击起始线段';
    }else{
      const range=fitContourPathIndices(fitRangeStart,index);
      if(!range.length){
        status.textContent='起止线段不在同一条连续 CAD 轮廓上，请重新选择';
        fitRangeStart=null;
      }else{
        const removedGreen=range.some(entityIndex=>selection.x.includes(entityIndex)||selection.y.includes(entityIndex));
        const added=range.filter(entityIndex=>!selection.fit.includes(entityIndex));
        range.forEach(entityIndex=>removeFromOtherSelections(entityIndex,'fit'));
        selection.fit.push(...added);
        const start=fitRangeStart;
        fitRangeStart=null;
        const summary=added.length
          ? '蓝色拟合轮廓已追加 '+added.length+' 条线段（起点实体 '+start+'，终点实体 '+index+'）'
          : '该连续轮廓已全部包含，无需重复追加';
        status.textContent=summary+'；当前共 '+selection.fit.length+' 条线段'+(removedGreen?'；重合的绿色边已取消，请重新选择':'')+'。可继续点击下一组起止线段追加。';
      }
    }
  }else{
    removeFromOtherSelections(index,mode); selection[mode]=[index];
    status.textContent=(mode==='x'?'绿色 X 边':'绿色 Y 边')+'已选择实体 '+index;
  }
  event.preventDefault(); event.stopPropagation(); drawSelections(); update();
}
function drawSelections(){
  if(!svg)return;
  svg.querySelectorAll('.cad-entity').forEach(entity=>{
    const index=Number(entity.dataset.index);
    entity.classList.toggle('selected-x',selection.x.includes(index));
    entity.classList.toggle('selected-y',selection.y.includes(index));
    entity.classList.toggle('selected-fit',selection.fit.includes(index));
    entity.classList.toggle('fit-range-start',fitRangeStart===index);
  });
}
function update(){
  const labels={x:'绿色 X 边',y:'绿色 Y 边',fit:'蓝色拟合轮廓'};
  list.innerHTML=Object.entries(selection).map(([key,values])=>'<div class="region-item"><b>'+labels[key]+'</b>：'+(values.length?'实体 '+values.join(', '):'未选择')+'</div>').join('');
  hidden.value=JSON.stringify(selection);
}
form.addEventListener('submit',event=>{
  if(pointcloudPreviewBusy||cadDirectionPreviewBusy){event.preventDefault();alert('点云预览仍在处理中，请等待 CAD 方向预览完成');return;}
  if(!pointcloudInput.files.length&&!pointcloudPreviewToken.value){event.preventDefault();alert('请先选择点云文件，或接收本项目拼接流程的最终侧面截面');return;}
  if(!pointcloudPreviewToken.value){event.preventDefault();alert('请先等待点云侧面预览生成成功，再开始分析');return;}
  if(!selection.x.length||!selection.y.length||!selection.fit.length){event.preventDefault();alert('请先点击选择绿色 X 边、绿色 Y 边和蓝色拟合轮廓');return;}
  hidden.value=JSON.stringify(selection);
});
if(initialExternalPreview&&initialExternalPreview.token){
  pointcloudPreviewToken.value=initialExternalPreview.token;
  pointcloudPreviewTitle.textContent='最终侧面截面';
  pointcloudPreviewImage.src=initialExternalPreview.image_url+'?t='+Date.now();
  pointcloudPreviewImage.classList.remove('hidden');
  pointcloudPreviewPlaceholder.classList.add('hidden');
  pointcloudFiles.textContent='已接收本项目拼接流程的最终清噪侧面截面，无需重复上传点云。';
  setPreviewStatus('截面已接收。请选择 DXF，随后继续原有 CAD 选线、全轮廓配准和偏差分析。');
  pointcloudPreviewMeta.textContent='来源：本项目拼接流程；截面点数：'+(initialExternalPreview.section_points||'未统计')+'；原始扫描数：'+(initialExternalPreview.source_file_count||'未统计');
}else{
  clearPointcloudPreview();
}
update();
</script>
</body>
</html>'''

def preview_svg(path):
    doc = ezdxf.readfile(str(path))
    entities = list(doc.modelspace())
    items = []
    all_points = []
    for index, entity in enumerate(entities):
        points = sample_entity(entity)
        if len(points):
            items.append((index, points))
            all_points.append(points)
    if not all_points:
        raise RuntimeError('DXF 中没有可预览的线或圆弧')
    points = np.vstack(all_points)
    minx, miny = points.min(0)
    maxx, maxy = points.max(0)
    pad = max(maxx - minx, maxy - miny) * .05 or 1
    minx -= pad
    maxx += pad
    miny -= pad
    maxy += pad
    width = maxx - minx
    height = maxy - miny
    visible_width = 3.0
    paths = []
    for index, line in items:
        d = 'M ' + ' L '.join(f'{point[0]:.5f},{point[1]:.5f}' for point in line)
        start = line[0]
        end = line[-1]
        length = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
        paths.append(
            f'<path class="cad-entity" data-index="{index}" '
            f'data-start-x="{start[0]:.12g}" data-start-y="{start[1]:.12g}" '
            f'data-end-x="{end[0]:.12g}" data-end-y="{end[1]:.12g}" '
            f'data-length="{length:.12g}" d="{d}" fill="none" '
            f'stroke="#111827" stroke-width="{visible_width:.1f}" stroke-linecap="round" '
            f'stroke-linejoin="round" vector-effect="non-scaling-stroke" pointer-events="stroke" tabindex="0">'
            f'<title>CAD 实体 {index}</title></path>'
        )
    flip_y = miny + maxy
    connect_tolerance = max(1e-6, max(width, height) * 1e-8)
    return f'<svg xmlns="http://www.w3.org/2000/svg" data-flip-y="{flip_y}" data-connect-tolerance="{connect_tolerance:.12g}" viewBox="{minx} {miny} {width} {height}" preserveAspectRatio="xMidYMid meet"><g transform="translate(0 {flip_y}) scale(1 -1)">' + ''.join(paths) + '</g></svg>'

def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def validate_selected_entities(path, selected):
    if not isinstance(selected, dict):
        raise RuntimeError('CAD 线段选择数据无效，请重新点击线段')
    document = ezdxf.readfile(str(path))
    entities = list(document.modelspace())
    labels = {'x': '绿色 X 边', 'y': '绿色 Y 边', 'fit': '蓝色拟合轮廓'}
    result = {}
    for key in ('x', 'y', 'fit'):
        values = selected.get(key)
        if not isinstance(values, list):
            raise RuntimeError(labels[key] + '没有有效的实体编号')
        normalized = []
        for raw_index in values:
            if isinstance(raw_index, bool):
                raise RuntimeError(labels[key] + '实体编号无效')
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                raise RuntimeError(labels[key] + '实体编号无效：' + str(raw_index))
            if isinstance(raw_index, float) and not raw_index.is_integer():
                raise RuntimeError(labels[key] + '实体编号必须是整数：' + str(raw_index))
            if index < 0 or index >= len(entities):
                raise RuntimeError(labels[key] + '实体编号超出 DXF 范围：' + str(index))
            try:
                points = sample_entity(entities[index])
            except Exception as exc:
                raise RuntimeError(labels[key] + '实体 ' + str(index) + ' 无法采样：' + str(exc))
            if not len(points):
                raise RuntimeError(labels[key] + '实体 ' + str(index) + ' 不是可处理的线或圆弧')
            if index not in normalized:
                normalized.append(index)
        if not normalized:
            raise RuntimeError('请至少选择一条' + labels[key])
        if key in ('x', 'y') and len(normalized) != 1:
            raise RuntimeError(labels[key] + '只能选择一条 CAD 线段')
        result[key] = normalized
    if result['x'][0] == result['y'][0]:
        raise RuntimeError('绿色 X 边和绿色 Y 边不能选择同一条 CAD 线段')
    overlapping = sorted((set(result['x']) | set(result['y'])) & set(result['fit']))
    if overlapping:
        raise RuntimeError('绿色边不能同时作为蓝色拟合轮廓：实体 ' + ', '.join(map(str, overlapping)))
    return result

def select_entities(path, regions):
    document = ezdxf.readfile(str(path))
    entities = list(document.modelspace())
    result = {}
    for key in ('x', 'y', 'fit'):
        region = regions.get(key)
        if not region:
            raise RuntimeError('缺少 CAD 绘制区域：' + key)
        scores = []
        if region.get('type') == 'polygon':
            polygon = [(float(point['x']), float(point['y'])) for point in region.get('points', [])]
            if len(polygon) < 3:
                raise RuntimeError('蓝色拟合套索无效，请重新绘制')
            for index, entity in enumerate(entities):
                points = sample_entity(entity)
                if not len(points):
                    continue
                inside_ratio = sum(point_in_polygon(point, polygon) for point in points) / len(points)
                if inside_ratio >= 0.80:
                    scores.append((inside_ratio, index))
            found = [index for _, index in scores]
        else:
            xmin = min(float(region['x1']), float(region['x2']))
            xmax = max(float(region['x1']), float(region['x2']))
            ymin = min(float(region['y1']), float(region['y2']))
            ymax = max(float(region['y1']), float(region['y2']))
            for index, entity in enumerate(entities):
                points = sample_entity(entity)
                if not len(points):
                    continue
                inside = (points[:, 0] >= xmin) & (points[:, 0] <= xmax) & (points[:, 1] >= ymin) & (points[:, 1] <= ymax)
                inside_ratio = float(np.mean(inside))
                if inside_ratio > 0:
                    scores.append((inside_ratio, index))
            found = [max(scores)[1]] if scores else []
        if not found:
            raise RuntimeError(('套索' if key == 'fit' else '矩形') + '区域内没有识别到 CAD 实体：' + key)
        result[key] = found
    return result


def prepare_uploaded_pointclouds(paths, run_folder):
    if len(paths) == 1:
        return paths[0]
    if len(paths) != 5:
        raise RuntimeError('原始 Media 点云自动拼接目前需要 5 个扫描文件；如果只有一个文件，请上传已处理的截面 PLY。')
    import process_media_five_clouds as pipeline
    input_folder = run_folder / 'raw_pointclouds'
    output_folder = run_folder / 'media_processing'
    denoised_folder = output_folder / 'denoised_inputs'
    section_folder = output_folder / 'vertical_section'
    input_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)
    denoised_folder.mkdir(parents=True, exist_ok=True)
    section_folder.mkdir(parents=True, exist_ok=True)
    ordered_paths = sorted(paths, key=lambda item: item.name.lower())
    pipeline_paths = []
    for index, source in enumerate(ordered_paths):
        target = input_folder / f'pointcloud_{index:02d}_{source.name}'
        shutil.copy2(source, target)
        pipeline_paths.append(target)
    pipeline.INPUT = input_folder
    pipeline.OUTPUT = output_folder
    pipeline.DENOISED = denoised_folder
    pipeline.SECTION = section_folder
    pipeline.main()
    section_path = section_folder / 'vertical_section_points.ply'
    if not section_path.exists():
        raise RuntimeError('多点云处理完成，但没有生成垂直截面点云')
    return section_path

def _preview_folder(token):
    if not isinstance(token, str) or not re.fullmatch(r'[0-9a-f]{32}', token):
        raise RuntimeError('点云预览标识无效，请重新选择点云')
    root = POINTCLOUD_PREVIEW_ROOT.resolve()
    folder = (POINTCLOUD_PREVIEW_ROOT / token).resolve()
    if folder.parent != root or not folder.is_dir():
        raise RuntimeError('点云预览已过期，请重新选择点云')
    return folder


def _file_manifest(paths):
    entries = []
    for path in sorted(paths, key=lambda item: item.name.lower()):
        digest = hashlib.sha256()
        with path.open('rb') as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        entries.append({'name': path.name, 'size': path.stat().st_size, 'sha256': digest.hexdigest()})
    return entries


def _save_pointcloud_uploads(files, folder):
    if not files or any(not item.filename for item in files):
        raise RuntimeError('没有收到有效的点云文件')
    upload_folder = folder / 'uploads'
    upload_folder.mkdir(parents=True, exist_ok=True)
    ordered_files = sorted(files, key=lambda item: secure_filename(item.filename).lower())
    paths = []
    for index, item in enumerate(ordered_files):
        safe_name = secure_filename(item.filename) or f'pointcloud_{index}.ply'
        path = upload_folder / f'{index:02d}_{safe_name}'
        item.save(path)
        paths.append(path)
    return paths


def _height_colormap(values):
    low, high = np.percentile(values, [1.0, 99.0])
    normalized = np.clip((values - low) / max(high - low, 1e-12), 0.0, 1.0)
    return plt.get_cmap('turbo')(normalized)


def _contract_section_profile(pointcloud_path, processing_report_path=None):
    cloud = o3d.io.read_point_cloud(str(pointcloud_path))
    points = np.asarray(cloud.points)
    if points.ndim != 2 or points.shape[1] != 3:
        return None
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    if len(points) < 3:
        return None
    processing_report = None
    if processing_report_path and Path(processing_report_path).exists():
        try:
            processing_report = json.loads(Path(processing_report_path).read_text(encoding='utf-8'))
        except (OSError, ValueError, TypeError):
            processing_report = None
    profile = section_profile_from_points(points, processing_report)
    if profile is None:
        return None
    section = processing_report.get('vertical_section', {}) if isinstance(processing_report, dict) else {}
    if section.get('coordinate_system') == '26xV01Q_section_local':
        profile_axis_xy = np.array([1.0, 0.0], dtype=float)
    else:
        reported_axis = section.get('profile_axis_xy')
        if not isinstance(reported_axis, (list, tuple)):
            axis = np.asarray(section.get('axis_xy', []), dtype=float)
            reported_axis = [-axis[1], axis[0]] if axis.shape == (2,) else [1.0, 0.0]
        profile_axis_xy = np.asarray(reported_axis[:2], dtype=float)
    profile_axis_norm = np.linalg.norm(profile_axis_xy)
    if not np.isfinite(profile_axis_norm) or profile_axis_norm < 1e-12:
        profile_axis_xy = np.array([1.0, 0.0], dtype=float)
    else:
        profile_axis_xy = profile_axis_xy / profile_axis_norm
    return profile, int(len(points)), profile_axis_xy


def _render_contract_section_preview(profile, image_path):
    profile = np.asarray(profile, dtype=float)
    if len(profile) > 250000:
        random_generator = np.random.default_rng(20260824)
        profile = profile[random_generator.choice(len(profile), 250000, replace=False)]
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if 'Microsoft YaHei' in {font.name for font in font_manager.fontManager.ttflist}:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    figure, axis = plt.subplots(figsize=(8.5, 7.5), dpi=170)
    axis.scatter(
        profile[:, 0],
        profile[:, 1],
        c=_height_colormap(profile[:, 1]),
        s=1.0,
        linewidths=0,
        rasterized=True,
    )
    axis.set_title('融合处理后的点云侧面参考图')
    axis.set_xlabel('横向距离')
    axis.set_ylabel('截面竖向坐标')
    axis.set_aspect('equal', adjustable='datalim')
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(image_path, bbox_inches='tight')
    plt.close(figure)


def create_pointcloud_side_preview(pointcloud_path, image_path, processing_report_path=None):
    contract = _contract_section_profile(pointcloud_path, processing_report_path)
    if contract is not None:
        profile, point_count, profile_axis_xy = contract
        _render_contract_section_preview(profile, image_path)
        return {
            'point_count': point_count,
            'preview_point_count': int(min(len(profile), 250000)),
            'profile_axis_angle_deg': float(np.degrees(np.arctan2(profile_axis_xy[1], profile_axis_xy[0]))),
        }
    cloud = o3d.io.read_point_cloud(str(pointcloud_path))
    points = np.asarray(cloud.points)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 3:
        raise RuntimeError('点云没有足够的三维点，无法生成侧面预览')
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    if len(points) < 3:
        raise RuntimeError('点云中没有有效的三维坐标')

    main_axis_xy = None
    center_xy = None
    profile_axis_xy = None
    if processing_report_path and Path(processing_report_path).exists():
        try:
            processing_report = json.loads(Path(processing_report_path).read_text(encoding='utf-8'))
            section = processing_report.get('vertical_section', {})
            if section.get('axis_xy') and section.get('plane_center_xy'):
                main_axis_xy = np.asarray(section['axis_xy'], dtype=float)
                center_xy = np.asarray(section['plane_center_xy'], dtype=float)
        except (OSError, ValueError, TypeError):
            main_axis_xy = None
            center_xy = None
    if main_axis_xy is not None and center_xy is not None and np.isfinite(main_axis_xy).all() and np.isfinite(center_xy).all():
        axis_norm = np.linalg.norm(main_axis_xy)
        if not np.isfinite(axis_norm) or axis_norm < 1e-12:
            raise RuntimeError('无法确定点云侧面投影方向')
        main_axis_xy = main_axis_xy / axis_norm
        profile_axis_xy = np.array([-main_axis_xy[1], main_axis_xy[0]])
    else:
        center_xy = np.median(points[:, :2], axis=0)
        covariance = np.cov((points[:, :2] - center_xy).T)
        eigenvalues, directions = np.linalg.eigh(covariance)
        major_axis_xy = directions[:, 1]
        major_norm = np.linalg.norm(major_axis_xy)
        if not np.isfinite(major_norm) or major_norm < 1e-12:
            raise RuntimeError('无法确定点云侧面投影方向')
        major_axis_xy = major_axis_xy / major_norm
        thin_ratio = float(eigenvalues[0] / max(eigenvalues[1], 1e-12))
        if thin_ratio < 0.01:
            profile_axis_xy = major_axis_xy
        else:
            profile_axis_xy = np.array([-major_axis_xy[1], major_axis_xy[0]])

    profile = np.column_stack(((points[:, :2] - center_xy) @ profile_axis_xy, points[:, 2]))
    if len(profile) > 250000:
        random_generator = np.random.default_rng(20260824)
        sampled_indices = random_generator.choice(len(profile), 250000, replace=False)
        profile = profile[sampled_indices]
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if 'Microsoft YaHei' in {font.name for font in font_manager.fontManager.ttflist}:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    figure, axis = plt.subplots(figsize=(8.5, 7.5), dpi=170)
    axis.scatter(profile[:, 0], profile[:, 1], c=_height_colormap(profile[:, 1]), s=1.0, linewidths=0, rasterized=True)
    axis.set_title('融合处理后的点云侧面参考图')
    axis.set_xlabel('横向距离')
    axis.set_ylabel('Z 高度')
    axis.set_aspect('equal', adjustable='datalim')
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(image_path, bbox_inches='tight')
    plt.close(figure)
    return {
        'point_count': int(len(points)),
        'preview_point_count': int(len(profile)),
        'profile_axis_angle_deg': float(np.degrees(np.arctan2(profile_axis_xy[1], profile_axis_xy[0]))),
    }

def _pointcloud_side_profile_for_cad_direction(pointcloud_path, processing_report_path=None):
    contract = _contract_section_profile(pointcloud_path, processing_report_path)
    if contract is not None:
        profile, point_count, _ = contract
        if len(profile) > 250000:
            random_generator = np.random.default_rng(20260824)
            profile = profile[random_generator.choice(len(profile), 250000, replace=False)]
        return profile, point_count
    cloud = o3d.io.read_point_cloud(str(pointcloud_path))
    points = np.asarray(cloud.points)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 3:
        raise RuntimeError('点云没有足够的三维点，无法生成 CAD 方向预览')
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        raise RuntimeError('点云中没有有效的三维坐标')

    main_axis_xy = None
    center_xy = None
    if processing_report_path and Path(processing_report_path).exists():
        try:
            processing_report = json.loads(Path(processing_report_path).read_text(encoding='utf-8'))
            section = processing_report.get('vertical_section', {})
            if section.get('axis_xy') and section.get('plane_center_xy'):
                main_axis_xy = np.asarray(section['axis_xy'], dtype=float)
                center_xy = np.asarray(section['plane_center_xy'], dtype=float)
        except (OSError, ValueError, TypeError):
            main_axis_xy = None
            center_xy = None
    if main_axis_xy is not None and center_xy is not None and np.isfinite(main_axis_xy).all() and np.isfinite(center_xy).all():
        axis_norm = np.linalg.norm(main_axis_xy)
        if not np.isfinite(axis_norm) or axis_norm < 1e-12:
            raise RuntimeError('无法确定点云侧面投影方向')
        main_axis_xy = main_axis_xy / axis_norm
        profile_axis_xy = np.array([-main_axis_xy[1], main_axis_xy[0]])
    else:
        center_xy = np.median(points[:, :2], axis=0)
        covariance = np.cov((points[:, :2] - center_xy).T)
        eigenvalues, directions = np.linalg.eigh(covariance)
        major_axis_xy = directions[:, 1]
        major_norm = np.linalg.norm(major_axis_xy)
        if not np.isfinite(major_norm) or major_norm < 1e-12:
            raise RuntimeError('无法确定点云侧面投影方向')
        major_axis_xy = major_axis_xy / major_norm
        thin_ratio = float(eigenvalues[0] / max(eigenvalues[1], 1e-12))
        profile_axis_xy = major_axis_xy if thin_ratio < 0.01 else np.array([-major_axis_xy[1], major_axis_xy[0]])

    profile = np.column_stack(((points[:, :2] - center_xy) @ profile_axis_xy, points[:, 2]))
    if len(profile) > 250000:
        random_generator = np.random.default_rng(20260824)
        profile = profile[random_generator.choice(len(profile), 250000, replace=False)]
    return profile, int(len(points))


def _normalize_preview_fit_entities(fit_entities, entities):
    if not fit_entities:
        return []
    if not isinstance(fit_entities, list):
        raise RuntimeError('蓝色拟合线段数据无效，请重新选择')
    result = []
    for raw_index in fit_entities:
        if isinstance(raw_index, bool):
            raise RuntimeError('蓝色拟合线段编号无效')
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise RuntimeError('蓝色拟合线段编号无效：' + str(raw_index)) from exc
        if isinstance(raw_index, float) and not raw_index.is_integer():
            raise RuntimeError('蓝色拟合线段编号必须是整数：' + str(raw_index))
        if index < 0 or index >= len(entities):
            raise RuntimeError('蓝色拟合线段编号超出 DXF 范围：' + str(index))
        if not len(sample_entity(entities[index])):
            raise RuntimeError('蓝色拟合实体不是可处理的线或圆弧：' + str(index))
        if index not in result:
            result.append(index)
    return result


def create_cad_direction_preview(pointcloud_path, dxf_path, image_path, processing_report_path=None, fit_entities=None):
    profile, point_count = _pointcloud_side_profile_for_cad_direction(pointcloud_path, processing_report_path)
    document = ezdxf.readfile(str(dxf_path))
    entities = list(document.modelspace())
    cad_items = []
    all_parts = []
    for index, entity in enumerate(entities):
        geometry = sample_entity(entity)
        if len(geometry):
            cad_items.append((index, geometry))
            all_parts.append(geometry)
    if not all_parts:
        raise RuntimeError('DXF 中没有可用于方向匹配的线或圆弧')
    selected_fit_indices = _normalize_preview_fit_entities(fit_entities, entities)
    cad_points = np.vstack(all_parts)
    selected_anchor = None
    if selected_fit_indices:
        fit_parts = [geometry for index, geometry in cad_items if index in set(selected_fit_indices)]
        if not fit_parts:
            raise RuntimeError('\u6240\u9009\u84dd\u8272\u62df\u5408\u7ebf\u6bb5\u6ca1\u6709\u53ef\u7528\u51e0\u4f55\u6570\u636e')
        selected_anchor = np.vstack(fit_parts)
        alignment_source = 'fixed_pointcloud_with_selected_blue_anchor'
    else:
        alignment_source = 'fixed_pointcloud'

    fit_center, mirror, fit_parameters, distances, fit_diagnostics = fit_profile_global(
        profile,
        cad_points,
        selected_anchor,
    )
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    height_low, height_high = np.percentile(profile[:, 1], [1.0, 99.0])
    if not np.isfinite(height_low) or not np.isfinite(height_high):
        raise RuntimeError('点云高度数据无效，无法生成高度彩色渲染图')
    if height_high - height_low < 1e-12:
        height_high = height_low + 1.0
    save_height_colored_cad_overlay_image(
        image_path,
        profile,
        profile[:, 1],
        entities,
        fit_indices=selected_fit_indices,
        cad_transform=(fit_center, mirror, fit_parameters[0], fit_parameters[1:]),
    )
    return {
        'point_count': point_count,
        'cad_to_pointcloud_angle_deg': float(np.degrees(fit_parameters[0])),
        'pointcloud_transformed': False,
        'mirror_applied': bool(mirror < 0),
        'mean_residual': float(np.mean(distances)),
        'alignment_source': alignment_source,
        'selected_fit_count': len(selected_fit_indices),
        'fit_method': fit_diagnostics['method'],
        'global_cad_to_pointcloud_mean': fit_diagnostics['global_cad_to_pointcloud_mean'],
        'global_pointcloud_to_cad_mean': fit_diagnostics['global_pointcloud_to_cad_mean'],
        'height_low': float(height_low),
        'height_high': float(height_high),
    }


def _preview_analysis_pointcloud(token):
    folder = _preview_folder(token)
    manifest_path = folder / 'manifest.json'
    if not manifest_path.exists():
        raise RuntimeError('点云预览信息不存在，请重新选择点云')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    relative_pointcloud = Path(manifest.get('analysis_pointcloud', ''))
    pointcloud_path = (folder / relative_pointcloud).resolve()
    if folder.resolve() not in pointcloud_path.parents or not pointcloud_path.exists():
        raise RuntimeError('点云预览文件不完整，请重新选择点云')
    processing_report_path = folder / 'media_processing' / 'processing_report.json'
    return folder, pointcloud_path, processing_report_path


def _processing_summary(processing_report):
    processing = processing_report.get('processing')
    if not isinstance(processing, dict):
        return None
    timing = processing.get('timing_seconds')
    accelerator = processing.get('accelerator')
    if not isinstance(timing, dict):
        timing = {}
    if not isinstance(accelerator, dict):
        accelerator = {}
    gpu = accelerator.get('gpu')
    if not isinstance(gpu, dict):
        gpu = {}
    return {
        'total_seconds': timing.get('total'),
        'registration_seconds': timing.get('registration'),
        'fusion_seconds': timing.get('fusion'),
        'fusion_backend': accelerator.get('backend'),
        'gpu_used': bool(accelerator.get('used')),
        'gpu_device': gpu.get('device_name'),
    }

def build_pointcloud_preview(paths, folder, token):
    pointcloud_path = prepare_uploaded_pointclouds(paths, folder)
    processing_report_path = folder / 'media_processing' / 'processing_report.json'
    source_image = None
    section_points = None
    processing = None
    mode = 'single_cloud'
    if len(paths) == 5:
        mode = 'five_cloud_denoise_registration_vertical_section'
        source_image = folder / 'media_processing' / 'vertical_section' / 'vertical_section_only.png'
        if not source_image.exists():
            raise RuntimeError('多点云处理完成，但没有生成点云侧面截图')
        if processing_report_path.exists():
            try:
                processing_report = json.loads(processing_report_path.read_text(encoding='utf-8'))
                section_points = processing_report.get('vertical_section', {}).get('section_points')
                processing = _processing_summary(processing_report)
            except (OSError, ValueError, TypeError):
                section_points = None
                processing = None
    else:
        source_image = folder / 'pointcloud_side_preview.png'
        preview_info = create_pointcloud_side_preview(pointcloud_path, source_image, processing_report_path)
        section_points = preview_info['point_count']
    side_image = folder / 'pointcloud_side_preview.png'
    if source_image.resolve() != side_image.resolve():
        shutil.copy2(source_image, side_image)
    manifest = {
        'token': token,
        'files': _file_manifest(paths),
        'analysis_pointcloud': str(Path(pointcloud_path).relative_to(folder)),
        'mode': mode,
        'section_points': int(section_points) if section_points is not None else None,
        'processing': processing,
    }
    (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest

def reuse_pointcloud_preview(token, paths, run_folder):
    preview_folder = _preview_folder(token)
    manifest_path = preview_folder / 'manifest.json'
    if not manifest_path.exists():
        raise RuntimeError('点云预览信息不存在，请重新选择点云')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    external_section = manifest.get('mode') == EXTERNAL_SECTION_MODE
    if not external_section and manifest.get('files') != _file_manifest(paths):
        raise RuntimeError('点云文件已更换，请重新选择点云并等待侧面预览完成')
    relative_pointcloud = Path(manifest.get('analysis_pointcloud', ''))
    source_pointcloud = (preview_folder / relative_pointcloud).resolve()
    if preview_folder.resolve() not in source_pointcloud.parents or not source_pointcloud.exists():
        raise RuntimeError('点云预览文件不完整，请重新选择点云')
    if manifest.get('mode') in ('five_cloud_denoise_registration_vertical_section', EXTERNAL_SECTION_MODE):
        target_processing = run_folder / 'media_processing'
        target_section = target_processing / 'vertical_section'
        target_section.mkdir(parents=True, exist_ok=True)
        source_processing = preview_folder / 'media_processing'
        source_section = source_processing / 'vertical_section'
        shutil.copy2(source_section / 'vertical_section_points.ply', target_section / 'vertical_section_points.ply')
        shutil.copy2(source_processing / 'processing_report.json', target_processing / 'processing_report.json')
        pointcloud_path = target_section / 'vertical_section_points.ply'
    else:
        pointcloud_path = paths[0]
    return pointcloud_path, manifest


def external_preview_payload(token):
    folder = _preview_folder(token)
    manifest_path = folder / 'manifest.json'
    if not manifest_path.exists():
        raise RuntimeError('截面交接信息不存在，请重新生成最终侧面截面')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('mode') != EXTERNAL_SECTION_MODE:
        raise RuntimeError('该预览不是本项目生成的最终截面交接数据')
    relative_pointcloud = Path(manifest.get('analysis_pointcloud', ''))
    pointcloud_path = (folder / relative_pointcloud).resolve()
    if folder.resolve() not in pointcloud_path.parents or not pointcloud_path.is_file():
        raise RuntimeError('最终截面交接文件不完整，请重新生成')
    return {
        'token': token,
        'image_url': url_for('pointcloud_preview_artifact', token=token),
        'section_points': manifest.get('section_points'),
        'source_file_count': manifest.get('source_file_count'),
        'mode': manifest.get('mode'),
    }

@app.route('/pointcloud_preview', methods=['POST'])
def pointcloud_preview():
    pointclouds = request.files.getlist('pointcloud')
    if not pointclouds or any(not item.filename for item in pointclouds):
        return jsonify(error='请至少选择一个点云文件'), 400
    if len(pointclouds) not in (1, 5):
        return jsonify(error='请上传 1 个已处理截面 PLY，或一次选择 5 个原始 Media 点云'), 400
    token = uuid.uuid4().hex
    folder = POINTCLOUD_PREVIEW_ROOT / token
    folder.mkdir(parents=True, exist_ok=False)
    try:
        paths = _save_pointcloud_uploads(pointclouds, folder)
        manifest = build_pointcloud_preview(paths, folder, token)
        return jsonify(
            token=token,
            image_url=url_for('pointcloud_preview_artifact', token=token),
            file_count=len(paths),
            mode=manifest['mode'],
            section_points=manifest.get('section_points'),
            processing=manifest.get('processing'),
        )
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        return jsonify(error=str(exc)), 400


@app.route('/pointcloud_preview_artifact/<token>')
def pointcloud_preview_artifact(token):
    try:
        folder = _preview_folder(token)
    except RuntimeError:
        return Response('not found', status=404)
    path = folder / 'pointcloud_side_preview.png'
    if not path.exists():
        return Response('not found', status=404)
    return send_file(path, mimetype='image/png')

@app.route('/pointcloud_cad_direction_preview', methods=['POST'])
def pointcloud_cad_direction_preview():
    token = request.form.get('pointcloud_preview_token', '').strip()
    dxf_file = request.files.get('dxf')
    if not token:
        return jsonify(error='缺少点云预览标识，请重新选择点云'), 400
    if not dxf_file or not dxf_file.filename:
        return jsonify(error='没有收到 DXF 图纸'), 400
    try:
        folder, pointcloud_path, processing_report_path = _preview_analysis_pointcloud(token)
        raw_fit_entities = request.form.get('fit_entities', '').strip()
        fit_entities = None
        if raw_fit_entities:
            try:
                fit_entities = json.loads(raw_fit_entities)
            except json.JSONDecodeError as exc:
                raise RuntimeError('蓝色拟合线段数据无法解析，请重新选择') from exc
            if not isinstance(fit_entities, list):
                raise RuntimeError('蓝色拟合线段数据无效，请重新选择')
        dxf_bytes = dxf_file.read()
        if not dxf_bytes:
            raise RuntimeError('DXF 图纸为空')
        digest = hashlib.sha256(dxf_bytes).hexdigest()[:24]
        selection_digest = 'all'
        if fit_entities:
            selection_digest = 'fit_' + hashlib.sha256(json.dumps(fit_entities, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()[:16]
        dxf_path = folder / f'cad_direction_{digest}.dxf'
        image_path = folder / f'pointcloud_cad_height_comparison_v7_{digest}_{selection_digest}.png'
        metadata_path = folder / f'pointcloud_cad_height_comparison_v7_{digest}_{selection_digest}.json'
        if not dxf_path.exists():
            dxf_path.write_bytes(dxf_bytes)
        if image_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        else:
            metadata = create_cad_direction_preview(pointcloud_path, dxf_path, image_path, processing_report_path, fit_entities=fit_entities)
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        return jsonify(
            image_url=url_for('pointcloud_preview_named_artifact', token=token, name=image_path.name),
            cad_to_pointcloud_angle_deg=metadata['cad_to_pointcloud_angle_deg'],
            pointcloud_transformed=metadata.get('pointcloud_transformed', False),
            mirror_applied=metadata['mirror_applied'],
            alignment_source=metadata['alignment_source'],
            selected_fit_count=metadata['selected_fit_count'],
            height_low=metadata['height_low'],
            height_high=metadata['height_high'],
        )
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.route('/pointcloud_preview_artifact/<token>/<name>')
def pointcloud_preview_named_artifact(token, name):
    try:
        folder = _preview_folder(token)
    except RuntimeError:
        return Response('not found', status=404)
    safe_name = secure_filename(name)
    if safe_name != name:
        return Response('not found', status=404)
    path = folder / safe_name
    if not path.exists() or not path.is_file():
        return Response('not found', status=404)
    return send_file(path, mimetype='image/png')


@app.route('/cad_preview', methods=['POST'])
def cad_preview():
    file=request.files.get('dxf')
    if not file: return jsonify(error='没有收到 DXF'),400
    path=WORK/f'preview_{uuid.uuid4().hex}.dxf'; file.save(path)
    try: return jsonify(svg=preview_svg(path))
    except Exception as exc: return jsonify(error=str(exc)),400

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'GET':
        external_preview = None
        section_token = request.args.get('section_token', '').strip()
        if section_token:
            try:
                external_preview = external_preview_payload(section_token)
            except Exception as exc:
                flash('接收最终侧面截面失败：' + str(exc))
        return render_template_string(HTML, report=None, external_preview=external_preview)
    pointclouds = request.files.getlist('pointcloud')
    dx = request.files.get('dxf')
    preview_token = request.form.get('pointcloud_preview_token', '').strip()
    run = uuid.uuid4().hex
    folder = WORK / run
    folder.mkdir()
    valid_pointclouds = [item for item in pointclouds if item.filename]
    if not dx or not dx.filename or (not valid_pointclouds and not preview_token):
        shutil.rmtree(folder, ignore_errors=True)
        flash('请选择 DXF，并上传点云或接收本项目拼接流程的最终侧面截面。')
        return redirect(url_for('index'))
    if valid_pointclouds and len(valid_pointclouds) not in (1, 5):
        shutil.rmtree(folder, ignore_errors=True)
        flash('请上传 1 个已处理截面 PLY，或一次选择 5 个原始 Media 点云。')
        return redirect(url_for('index'))
    preview_reused = False
    preview_manifest = None
    try:
        paths = _save_pointcloud_uploads(valid_pointclouds, folder) if valid_pointclouds else []
        dxf_path = folder / (secure_filename(dx.filename) or 'drawing.dxf')
        dx.save(dxf_path)
        if preview_token:
            pointcloud_path, preview_manifest = reuse_pointcloud_preview(preview_token, paths, folder)
            preview_reused = True
        else:
            pointcloud_path = prepare_uploaded_pointclouds(paths, folder)
        selected_entities_payload = request.form.get('selected_entities', '').strip()
        if selected_entities_payload:
            try:
                selected_entities = json.loads(selected_entities_payload)
            except json.JSONDecodeError as exc:
                raise RuntimeError('CAD 线段选择数据无法解析，请重新点击线段') from exc
            groups = validate_selected_entities(dxf_path, selected_entities)
        else:
            selected_regions = json.loads(request.form.get('selected_regions', '{}'))
            groups = select_entities(dxf_path, selected_regions)
        report = analyze(
            pointcloud_path,
            dxf_path,
            folder,
            step=float(request.form.get('step', .05)),
            x_entities=groups['x'],
            y_entities=groups['y'],
            fit_entities=groups['fit'],
        )
        report['cad_selection'] = groups
        report['preprocessing'] = {
            'input_cloud_count': len(paths) or int((preview_manifest or {}).get('source_file_count') or 0),
            'input_clouds': [path.name for path in paths],
            'analysis_pointcloud': str(pointcloud_path),
            'mode': preview_manifest.get('mode') if preview_manifest else ('five_cloud_denoise_registration_vertical_section' if len(paths) == 5 else 'single_cloud'),
            'preview_reused': preview_reused,
        }
        (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        flash('分析失败：' + str(exc))
        return redirect(url_for('index'))
    return render_template_string(
        HTML,
        report=report,
        run_id=run,
        external_preview=None,
    )


@app.route('/health')
def health():
    return Response('ok', mimetype='text/plain')
@app.route('/artifact/<run>/<name>')
def artifact(run,name):
    path=(WORK/run/secure_filename(name)).resolve()
    if WORK.resolve() not in path.parents or not path.exists(): return Response('not found',status=404)
    return send_file(path,as_attachment=not name.endswith('.png'))

if __name__=='__main__':
    app.run(
        host='127.0.0.1',
        port=int(os.environ.get('PORT',5000)),
        debug=os.environ.get('FLASK_DEBUG') == '1',
        use_reloader=False,
    )








