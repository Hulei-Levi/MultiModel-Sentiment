#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按附件四样本编号生成分类、回归两张归因综合图。

用法（在 question3 目录）：
    ../.venv-align/bin/python -B plot_sample_waterfall.py 1
    ../.venv-align/bin/python -B plot_sample_waterfall.py 样本19
    ../.venv-align/bin/python -B plot_sample_waterfall.py --all

仅读取已有归因、WhisperX 对齐和媒体文件，不重新推理模型。
依赖：numpy、matplotlib、Pillow（现有 .venv-align 环境已具备）。
每张综合图含模态贡献瀑布图、任务各自的前三帧、三模态局部归因热图、
词语与原始媒体时间对应。默认导出 PDF、SVG、360 dpi PNG，以及两页 PDF。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

Q3_ROOT = Path(__file__).resolve().parent
ROOT = Q3_ROOT.parent
DEFAULT_COMPLETION = Q3_ROOT / "completion_20260926"
DEFAULT_REPORT = Q3_ROOT / "attachment4_all_20260925T152718_260737Z" / "analysis_report.json"
MODALITIES = ("text", "audio", "vision")
HEADS = ("classification", "regression")
Q3_ZH_ROOT = DEFAULT_COMPLETION / "localization"
_Q3_ZH_CATALOG = {}

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def q3_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)

_Q3_PANEL_AUDIT_SOURCE = "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"

def q3_alignment_auditor():
    import base64, zlib, types
    name="_q3_embedded_panel_auditor"
    if name not in sys.modules:
        module=types.ModuleType(name)
        sys.modules[name]=module
        source=zlib.decompress(base64.b64decode(_Q3_PANEL_AUDIT_SOURCE)).decode("utf-8")
        exec(compile(source,"<embedded-panel-auditor>","exec"),module.__dict__)
    return sys.modules[name].require_matplotlib_panel_alignment

_Q3_ZH_LABELS = {
    "text":"文本", "audio":"语音", "vision":"视觉",
    "classification":"情感极性", "regression":"情感强度",
    "negative":"负向", "neutral":"中性", "positive":"正向",
    "mean":"训练均值参考", "zero":"零参考", "all":"三模态联合",
    "correct":"分类正确", "incorrect":"分类错误", "high":"高", "middle":"中", "low":"低",
    "train":"训练集", "valid":"验证集", "test":"测试集",
}

def q3_zh(value):
    return _Q3_ZH_LABELS.get(value,value)

def q3_zh_catalog(kind):
    key=(str(Q3_ZH_ROOT),kind)
    if key not in _Q3_ZH_CATALOG:
        path=Q3_ZH_ROOT/(kind+"_zh.json")
        if not path.is_file():
            raise FileNotFoundError("缺少中文展示资源："+str(path))
        _Q3_ZH_CATALOG[key]=json.loads(path.read_text(encoding="utf-8"))
    return _Q3_ZH_CATALOG[key]

def q3_zh_case(record):
    sid=record.get("id",record.get("sample_id"))
    kind="validation" if sid.startswith("v") else "attachment4"
    data=q3_zh_catalog(kind)
    item=data.get("samples",data)[sid]
    if item["raw_text"]!=record["raw_text"]:
        raise AssertionError("中文译文与原始文本不匹配："+sid)
    return item

def q3_zh_word(item,word,bilingual=True):
    translated=item["word_zh"][word]
    return translated+"（"+word+"）" if bilingual and translated!=word else translated

def q3_plot_style():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    path=Q3_ZH_ROOT/"fonts"/"DroidSansFallbackFull.ttf"
    if not path.is_file():
        raise FileNotFoundError("缺少中文字体："+str(path))
    font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({
        "font.family":["DejaVu Sans","Droid Sans Fallback"],"font.sans-serif":["DejaVu Sans","Droid Sans Fallback"],
        "font.size":8,"axes.titlesize":9,"axes.labelsize":8,
        "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,
        "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
        "axes.spines.top":False,"axes.spines.right":False,"axes.linewidth":0.7,
        "legend.frameon":False,"savefig.facecolor":"white","figure.facecolor":"white"})
    return plt

def q3_figure_label(ax, letter):
    ax.annotate(letter,(0,1),xycoords="axes fraction",xytext=(-31,12),
                textcoords="offset points",fontweight="bold",fontsize=9,fontfamily="DejaVu Sans",
                ha="left",va="bottom")

def q3_plot_sample_overview(args, sample_id):
    """Draw separate target overviews from saved values and original media."""
    from PIL import Image
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle, Polygon
    destination,media_root,record,evidence,alignment,words,heat,source=q3_overview_source(args,sample_id)
    sid=record["id"]
    plt=q3_plot_style()
    modalities=list(MODALITIES)
    modality_colors=["#3B6E8C","#BF8845","#5C8D78"]
    translation=q3_zh_case(record)
    # 图中使用简短等义释义，完整中文解释仍保留在原始中文资源中。
    short_gloss={"与后面的“再见”连用，表示告别":"告别",
                 "与后面的“例子”连用，表示例如":"例如",
                 "与前面的介词连用，表示当然":"当然"}
    translation={**translation,"word_zh":{word:short_gloss.get(zh,zh)
                 for word,zh in translation["word_zh"].items()}}
    gloss={"Replacing":"更换","these":"这些","wear":"磨损","components":"部件","when":"当……时",
           "replacing":"更换","the":"该","timing":"正时","belt":"皮带","is":"是","essential":"至关重要",
           "to":"以；用于","ensuring":"确保","new":"新的","performs":"发挥性能","its":"其",
           "mileage":"使用里程","requirements":"要求"}
    labels=[]
    for column in source["columns"]:
        raw=column["label"]
        if column["kind"]=="special":
            labels.append(raw+"\n"+{"[CLS]":"起始标记","[SEP]":"结束标记"}.get(raw,"特殊标记"))
        elif column["kind"]=="word":
            gloss_zh=gloss.get(raw,translation["word_zh"].get(raw,""))
            labels.append(raw+("\n"+gloss_zh if gloss_zh and gloss_zh!=raw else ""))
        else:
            labels.append(raw+"\n标点")
    column_by_word={c["word_index"]:i for i,c in enumerate(source["columns"]) if c["kind"]=="word"}
    ncol=len(labels)
    cmap=plt.get_cmap("RdBu")
    start=float(source["video_start_s"]);end=float(source["video_end_s"])
    duration=end-start
    outputs=[]
    book_path=destination/(sid+"_attribution_overview.pdf")
    with PdfPages(book_path) as book:
        for hi,head in enumerate(HEADS):
            limit=source["heatmap_color_limits"][head][1]
            norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
            figure_width=max(13.0, ncol*20.5/(.89*72))
            fig=plt.figure(figsize=(figure_width,7.5))
            base=record["baselines"]["mean"]
            phi=np.asarray(base["phi"][hi],float)
            shares=np.asarray(base["shares"][hi],float)
            reference=float(source["reference_output"][hi])
            final=float(source["full_coalition_output"][hi])
            title_head="情感极性分类" if head=="classification" else "情感强度回归"
            main= q3_zh(modalities[int(np.argmax(shares))])
            polarity=("负向","中性","正向")[record["prediction"]["class_index"]]
            fig.text(.075,.920,f"预测：{polarity}    情感强度：{record['prediction']['regression']:+.3f}"
                     f"    主要参考模态：{main}（{100*max(shares):.2f}%）",fontsize=10)
            axes=[]
            water=fig.add_axes([.075,.660,.260,.210]);axes.append(water)
            water.set_title("模态贡献的加和分解",fontsize=10,pad=12)
            running=reference
            levels=[reference]
            water.bar(0,reference,bottom=0,width=.63,color="#AFB3AC",edgecolor="#848A84",linewidth=.7)
            water.annotate(f"{reference:+.4f}",(0,reference),xytext=(0,7 if reference>=0 else -10),
                           textcoords="offset points",ha="center",va="bottom" if reference>=0 else "top",fontsize=8.5)
            for mi,value in enumerate(phi):
                previous=running;running+=float(value);levels.append(running)
                water.bar(mi+1,float(value),bottom=previous,width=.63,
                          color=modality_colors[mi],edgecolor=modality_colors[mi],linewidth=.8)
                water.plot([mi+.315,mi+.685],[previous,previous],color="#909898",lw=.65,ls="--")
                water.annotate(f"{value:+.4f}",(mi+1,running),xytext=(0,7 if value>=0 else -10),
                               textcoords="offset points",ha="center",va="bottom" if value>=0 else "top",fontsize=8.5)
            water.bar(4,final,bottom=0,width=.63,color="#424A4D",edgecolor="#424A4D",linewidth=.7)
            water.plot([3.315,3.685],[final,final],color="#909898",lw=.65,ls="--")
            water.annotate(f"{final:+.4f}",(4,final),xytext=(0,7 if final>=0 else -10),
                           textcoords="offset points",ha="center",va="bottom" if final>=0 else "top",fontsize=8.5)
            low=min(0,*levels,final);high=max(0,*levels,final);span=max(high-low,1e-3)
            water.set_ylim(low-.25*span,high+.28*span)
            water.set_xlim(-.6,4.6)
            water.set_xticks(range(5),["参考\n输入"]+
                [q3_zh(m)+f"\n{100*shares[i]:.2f}%" for i,m in enumerate(modalities)]+["完整\n输入"])
            water.tick_params(axis="both",labelsize=8)
            winner=("负向","中性","正向")[record["prediction"]["class_index"]]
            runner=("负向","中性","正向")[record["prediction"]["runnerup_index"]]
            water.set_ylabel(f"{winner}相对{runner}的得分差" if hi==0 else "情感强度分数",fontsize=9)
            frame_grid=fig.add_gridspec(1,3,left=.405,right=.965,bottom=.648,top=.87,wspace=.10)
            frame_records=[]
            for j,item in enumerate(evidence["top_evidence"][head]["vision"]):
                asset,role,status_note=resolve_frame(item,evidence)
                ax=fig.add_subplot(frame_grid[0,j]);axes.append(ax)
                if asset:
                    with Image.open(media_root/asset["file"]) as image:
                        ax.imshow(np.asarray(image),interpolation="none")
                        ax.set_box_aspect(image.height/image.width)
                else:
                    ax.set_box_aspect(9/16)
                    ax.text(.5,.5,"暂无对应画面",ha="center",va="center",transform=ax.transAxes,fontsize=10)
                ax.set_anchor("N");ax.set_axis_off()
                frame_title=f"视觉关键帧 {j+1}" if role=="confirmed" else f"参考画面 {j+1}"
                ax.set_title(frame_title+("（"+status_note+"）" if status_note else ""),fontsize=9,pad=12)
                word_label=q3_zh_word(translation,item["word"])
                detail=(f"{asset['frame_pts_s']:.3f}秒｜第{asset['decoded_frame_index']}帧" if asset else "无时间定位")
                ax.text(.5,-.07,word_label+"\n"+detail,
                        transform=ax.transAxes,ha="center",va="top",fontsize=8.5,linespacing=1.5)
                frame_record={"rank":item["rank"],"word_index":item["word_index"],"word":item["word"],
                    "role":role,"display_note":status_note,"counts_as_confirmed_evidence":role=="confirmed"}
                if asset:
                    frame_record.update(asset=asset["file"],asset_sha256=sha256(media_root/asset["file"]),
                        frame_pts_s=asset["frame_pts_s"],frame_index=asset["decoded_frame_index"])
                frame_records.append(frame_record)
            axh=fig.add_axes([.075,.425,.89,.115]);axes.append(axh)
            mesh=axh.pcolormesh(np.arange(ncol+1),np.arange(4),heat[hi],cmap=cmap,norm=norm,
                               edgecolors="#FFFFFF",linewidth=.5,shading="flat")
            axh.set_xlim(0,ncol);axh.set_ylim(3,0)
            axh.set_xticks(np.arange(ncol)+.5,labels,rotation=90,ha="right",va="center",rotation_mode="anchor",fontsize=7.5)
            axh.set_yticks(np.arange(3)+.5,["文本","语音","视觉"],fontsize=9)
            axh.tick_params(axis="both",length=0,pad=6)
            for tick,color in zip(axh.get_yticklabels(),modality_colors):tick.set_color(color)
            for spine in axh.spines.values():spine.set_visible(False)
            for mi,m in enumerate(modalities):
                for item in evidence["top_evidence"][head][m]:
                    col=column_by_word[item["word_index"]]
                    axh.add_patch(Rectangle((col+.045,mi+.045),.91,.91,fill=False,
                                            edgecolor="#222B30",linewidth=1.3))
            axh.set_title("三模态局部归因",loc="left",fontsize=10,pad=12)
            cbax=fig.add_axes([.783,.568,.182,.012])
            colorbar=fig.colorbar(mesh,cax=cbax,orientation="horizontal")
            colorbar.set_ticks([-limit,0,limit],labels=[f"{-limit:.3f}","0",f"{limit:.3f}"])
            colorbar.ax.tick_params(labelsize=7.5,length=2,pad=2)
            colorbar.outline.set_linewidth(.4)

            # fig.text(.783,.591,"净贡献：红色为负，蓝色为正",fontsize=8)

            axt=fig.add_axes([.075,.115,.89,.150]);axes.append(axt)
            axt.set_xlim(start,end);axt.set_ylim(-.02,1.02)
            for wi,w in enumerate(words):
                if not interval_available(w): continue
                col=column_by_word[w["word_index"]]
                left=start+col/ncol*duration;right=start+(col+1)/ncol*duration
                axt.add_patch(Polygon([(left,1),(right,1),(w["end_s"],.61),(w["start_s"],.61)],
                                     closed=True,facecolor="#DFE9EF",edgecolor="white",linewidth=.55,alpha=.85 if w["status"]=="aligned" else .40))
                axt.add_patch(Rectangle((w["start_s"],.0),w["end_s"]-w["start_s"],.265,
                                       facecolor="#EDF2F4",edgecolor="white",linewidth=.55))
            for mi,m in enumerate(("audio","text")):
                band_y=.015 if m=="audio" else .145
                color=modality_colors[modalities.index(m)]
                for item in evidence["top_evidence"][head][m]:
                    if not interval_available(item): continue
                    aligned=item.get("alignment_status")=="aligned"
                    axt.add_patch(Rectangle((item["start_s"],band_y),
                                            item["end_s"]-item["start_s"],.105,
                                            facecolor=color+"30",edgecolor=color,linewidth=1.25,linestyle="-" if aligned else "--"))
            timed_frames=sorted([v for v in frame_records if "frame_pts_s" in v],key=lambda v:v["frame_pts_s"])
            clusters=[]
            for item in timed_frames:
                t_frame=item["frame_pts_s"]
                marker="v" if item["role"]=="confirmed" else "o"
                axt.scatter([t_frame],[.34],marker=marker,s=34,color=modality_colors[2],zorder=5)
                if clusters and (t_frame-clusters[-1][-1]["frame_pts_s"])/duration*figure_width*.89*72<20:
                    clusters[-1].append(item)
                else: clusters.append([item])
            for cluster in clusters:
                t_label=float(np.mean([v["frame_pts_s"] for v in cluster]))
                axt.text(t_label,.43,"、".join(str(v["rank"]) for v in cluster),
                         ha="center",va="bottom",fontsize=8.5,color="#356851",zorder=6)
            axt.set_yticks([.1975,.0675],["文本","语音"],fontsize=8)
            axt.tick_params(axis="y",length=0,pad=7)
            axt.xaxis.set_major_locator(__import__("matplotlib").ticker.MaxNLocator(nbins=12,integer=True))
            axt.tick_params(axis="x",labelsize=8)
            axt.set_xlabel("原始视频时间（秒）",fontsize=9,labelpad=5)
            for side in ["top","right","left"]:axt.spines[side].set_visible(False)
            axt.spines["bottom"].set_color("#98A5AB")
            fig.text(.075,.295,"WhisperX 词语时间对应",fontsize=10)
            # fig.text(.51,.295,"蓝框：文本前三｜橙框：语音前三｜绿标：视觉前三帧",fontsize=8.5)
            # target_note=(f"分类解释目标为“{winner}相对{runner}的得分差”；正贡献增大该差值，负贡献减小该差值。"
            #              if hi==0 else "回归解释目标为情感强度分数；正贡献提高分数，负贡献降低分数。")
            # fig.text(.075,.045,target_note+"同图三模态共用色标；两任务数值不直接比较。",fontsize=7.5)
            # fig.text(.075,.019,"起始、结束标记参与归因但不对应媒体时间；连带表示词语时间对应，不代表缓存特征的原始提取窗口。",fontsize=7.5)
            notes=[]
            if source["unrepresented_words"]:
                notes.append(f"仅展示进入模型的 {len(words)}/{source['raw_word_count']} 个词及其余有效位置")
            if any(v["role"] in ("raw_context_only","unlocalized_context") for v in frame_records):
                notes.append("视觉输入全零；上方画面仅为原始视频参考")
            elif any(v["role"]!="confirmed" for v in frame_records):
                notes.append("参考画面保留原排名；虚线区间表示候选时间对应")
            if notes:
                fig.text(.075,.025,"；".join(notes)+"。",fontsize=7.5,color="#52626A")
            prefix=destination/(sid+"_"+head+"_overview")
            fig.canvas.draw()
            require_matplotlib_panel_alignment=q3_alignment_auditor()
            alignment_report=require_matplotlib_panel_alignment(fig,json_out=prefix.with_suffix(".alignment.json"),
                tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=False,strict=True,
                axes=axes,panel_ids=list("abcdef"),row_groups=[["b","c","d"]],column_groups=[["e","f"]])
            fig.savefig(prefix.with_suffix(".pdf"))
            fig.savefig(prefix.with_suffix(".svg"))
            fig.savefig(prefix.with_suffix(".png"),dpi=360)
            book.savefig(fig)
            metadata={"sample_id":sid,"head":head,"size_inches":[figure_width,7.5],"dpi":360,
                      "source_sha256":sha256(__file__),"source_data":sid+"_overview_source_data.json",
                      "heatmap_columns":ncol,"heatmap_signed_net":heat[hi].tolist(),
                      "reference_output":reference,"signed_shapley":phi.tolist(),"full_output":final,
                      "frames":frame_records,"alignment_verdict":alignment_report["verdict"],
                      "rendered_text":[t.get_text() for t in fig.findobj(match=__import__("matplotlib").text.Text)
                                       if t.get_visible() and t.get_text()],
                      "formats":["pdf","svg","png"],"prediction_values_unchanged":True,
                      "image_processing":"original frames; no crop or enhancement",
                      "color_limits":source["heatmap_color_limits"][head],"special_positions_mapped_to_time":False}
            q3_write_json(prefix.with_suffix(".figure.json"),metadata)
            outputs.append({"head":head,"stem":prefix.name,"frames":frame_records})
            plt.close(fig)
    q3_write_json(destination/(sid+"_overview_manifest.json"),
        {"status":"rendered_pending_qa","sample_id":sid,"source_sha256":sha256(__file__),
         "source_data":sid+"_overview_source_data.json","outputs":outputs,"combined_pdf":book_path.name,
         "reproduce_command":f"{sys.executable} -B {Path(__file__).resolve()} {sid}",
         "previous_cards_modified":False})
    print(f"样本 {sid} 已生成分类、回归综合图：{destination}",flush=True)
    return outputs


def normalize_sample_id(value):
    """接受 1、01、样本1、样本01；拒绝范围外或混合数字。"""
    match = re.fullmatch(r"(?:样本\s*)?([0-9]{1,2})", str(value).strip())
    if not match or not 1 <= int(match.group(1)) <= 20:
        raise argparse.ArgumentTypeError("样本编号应为 1—20，例如 1、01 或 样本1。")
    return f"{int(match.group(1)):02d}"


def q3_overview_source(args, sample_id):
    """保留所有有效特征槽，核对词语聚合和精确 Shapley 分解。"""
    sid = normalize_sample_id(sample_id)
    completion = Path(args.completion_root).resolve()
    media_root = completion / "attachment4"
    destination = (Path(args.output_dir).resolve() / ("sample" + sid)).resolve()
    if not destination.is_relative_to(Q3_ROOT):
        raise ValueError("输出目录必须位于 question3 目录内。")
    report_path = Path(args.attachment4_report).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = {r["id"]: r for r in report["samples"]}
    if sid not in records:
        raise ValueError(f"已有归因报告中找不到样本 {sid}。")
    record = records[sid]
    evidence_path = media_root / "samples" / sid / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    alignment_path = media_root / evidence["alignment_file"]
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    # 未进入模型的截断后文本不参与归因；部分可见词仅展示其可见特征槽。
    words = [w for w in evidence["words"] if w["represented"]]
    valid = np.asarray(record["valid_mask"], bool)
    content = np.asarray(record["content_mask"], bool)
    with np.load(report_path.parent / "attributions.npz", allow_pickle=False) as z:
        arrays = {m: np.asarray(z[f"s{sid}_mean_{m}"], np.float64) for m in MODALITIES}
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("局部归因包含非有限值。")
    covered = np.zeros(len(valid), float)
    columns = []
    for w in words:
        slots, weights = w["token_slots"], w["slot_weights"]
        if not slots or len(slots) != len(weights):
            raise ValueError("词语特征槽与聚合权重不匹配。")
        for slot, weight in zip(slots, weights):
            if not valid[slot] or not content[slot] or weight < 0:
                raise ValueError("词语映射到了无效特征槽。")
            covered[slot] += weight
        columns.append({
            "label": w["word"], "kind": "word", "word_index": w["word_index"],
            "start_s": w["start_s"], "end_s": w["end_s"], "alignment_status": w["status"],
            "token_slots": slots, "slot_weights": weights, "raw_char_span": w["raw_char_span"],
            "full_word_represented": w["full_word_represented"],
        })
    if np.any(covered > 1 + 1e-10):
        raise ValueError("词语权重重复计入了同一个特征槽。")
    # 标点和特殊标记均保留；不为它们虚构媒体时间。
    for slot in np.flatnonzero(valid):
        remaining = 1.0 - covered[slot]
        if remaining > 1e-10:
            columns.append({
                "label": record["tokens"][slot],
                "kind": "special" if not content[slot] else "unmapped_token",
                "token_slots": [int(slot)], "slot_weights": [float(remaining)],
                "slot": int(slot),
            })
    columns.sort(key=lambda c: (min(c["token_slots"]), c.get("word_index", -1)))
    heat = np.zeros((2, 3, len(columns)), np.float64)
    masses = np.zeros_like(heat)
    word_by_id = {w["word_index"]: w for w in words}
    for hi, head in enumerate(HEADS):
        for mi, modality in enumerate(MODALITIES):
            a = arrays[modality][hi]
            if np.any(a[~valid]):
                raise ValueError("填充位置归因非零，不能静默忽略。")
            for ci, c in enumerate(columns):
                heat[hi, mi, ci] = sum(float(a[s].sum()) * v for s, v in zip(c["token_slots"], c["slot_weights"]))
                masses[hi, mi, ci] = sum(float(np.abs(a[s]).sum()) * v for s, v in zip(c["token_slots"], c["slot_weights"]))
                if c["kind"] == "word":
                    w = word_by_id[c["word_index"]]
                    np.testing.assert_allclose(heat[hi, mi, ci], w["attribution_net"][head][modality], rtol=1e-10, atol=1e-10)
                    np.testing.assert_allclose(masses[hi, mi, ci], w["attribution_mass"][head][modality], rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(heat[hi, mi].sum(), a.sum(), rtol=1e-10, atol=1e-10)
    base = record["baselines"]["mean"]
    reference = np.asarray(base["coalitions"][0], float)
    full = np.asarray(base["coalitions"][7], float)
    phi = np.asarray(base["phi"], float)
    if not all(np.isfinite(v).all() for v in [reference, full, phi]):
        raise ValueError("瀑布图输入包含非有限值。")
    np.testing.assert_allclose(reference + phi.sum(1), full, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(heat.sum(2), base["integrated_modality_totals"], rtol=1e-10, atol=1e-10)
    limits = [max(float(np.abs(heat[h]).max()), 1e-12) for h in range(2)]
    preview_path = media_root / "visual_review_previews.json"
    previews = json.loads(preview_path.read_text(encoding="utf-8"))["samples"].get(sid, []) if preview_path.exists() else []
    evidence["_preview_by_word"] = {v["word_index"]: v for v in previews}
    evidence["_word_by_id"] = word_by_id
    for head in HEADS:
        for modality in MODALITIES:
            for item in evidence["top_evidence"][head][modality]:
                if item["word_index"] not in word_by_id:
                    raise ValueError("关键证据指向未进入模型的词语。")
    files = [report_path, report_path.parent / "attributions.npz", evidence_path, alignment_path]
    if preview_path.exists():
        files.append(preview_path)
    source = {
        "sample_id": sid, "baseline": "training_mean", "prediction": record["prediction"],
        "reference_output": reference.tolist(), "full_coalition_output": full.tolist(),
        "signed_modality_shapley": phi.tolist(), "modality_shares": base["shares"],
        "columns": columns, "heatmap_signed_net": heat.tolist(), "absolute_attribution_mass": masses.tolist(),
        "heatmap_color_limits": {head: [-limits[h], limits[h]] for h, head in enumerate(HEADS)},
        "heatmap_modality_sums": heat.sum(2).tolist(),
        "shapley_completeness_residual": (heat.sum(2) - phi).tolist(),
        "source_feature_windows_recovered": False,
        "time_mapping": "WhisperX post-hoc word intervals; special and unmapped tokens have no media time",
        "top_evidence": evidence["top_evidence"],
        "source_files": {str(v): sha256(v) for v in files},
        "video_start_s": alignment["video_pts_s"][0],
        "video_end_s": alignment["video_frame_end_s"][-1],
        "padding_positions_omitted": np.flatnonzero(~valid).tolist(),
        "padding_attribution_exactly_zero": True,
        "raw_word_count": len(evidence["words"]), "represented_word_count": len(words),
        "unrepresented_words": [w["word_index"] for w in evidence["words"] if not w["represented"]],
        "unmapped_token_count": sum(c["kind"] == "unmapped_token" for c in columns),
    }
    destination.mkdir(parents=True, exist_ok=True)
    q3_write_json(destination / (sid + "_overview_source_data.json"), source)
    return destination, media_root, record, evidence, alignment, words, heat, source


def resolve_frame(item, evidence):
    """只取同一词语的原帧或既有预览，不跨词、跨任务替换排名。"""
    if item.get("asset"):
        return item["asset"], "confirmed", ""
    preview = evidence["_preview_by_word"].get(item["word_index"])
    if preview and preview.get("asset"):
        role = preview["role"]
        note = "原始参考画面" if role in ("raw_context_only", "unlocalized_context") else "时间对应待复核"
        return preview["asset"], role, note
    return None, "unavailable", "暂无对应画面"


def interval_available(item):
    start, end = item.get("start_s"), item.get("end_s")
    return start is not None and end is not None and np.isfinite([start, end]).all() and end > start


def main(argv=None):
    global Q3_ZH_ROOT
    parser = argparse.ArgumentParser(
        description="输入附件四样本编号，输出分类和回归两张完整归因综合图。",
        epilog="例：python plot_sample_waterfall.py 1；python plot_sample_waterfall.py 样本19",
    )
    parser.add_argument("sample", nargs="?", type=normalize_sample_id, help="样本编号：1—20、01 或 样本1")
    parser.add_argument("--all", action="store_true", help="依次生成全部 20 个样本")
    parser.add_argument("--completion-root", type=Path, default=DEFAULT_COMPLETION, help="已有解释结果目录")
    parser.add_argument("--attachment4-report", type=Path, default=DEFAULT_REPORT, help="已有归因报告")
    parser.add_argument("--output-dir", type=Path, default=Q3_ROOT / "waterfall_results", help="输出根目录，其下自动建立 sampleXX")
    parser.add_argument("--check-data", action="store_true", help="仅核对归因数据并保存源数据，不绘图")
    args = parser.parse_args(argv)
    if args.all and args.sample:
        parser.error("样本编号与 --all 只能选择一个。")
    if not args.all and args.sample is None:
        try:
            args.sample = normalize_sample_id(input("请输入附件四样本编号（1—20）："))
        except (EOFError, argparse.ArgumentTypeError) as exc:
            parser.error(str(exc))
    Q3_ZH_ROOT = args.completion_root.resolve() / "localization"
    samples = [f"{i:02d}" for i in range(1, 21)] if args.all else [args.sample]
    for sid in samples:
        if args.check_data:
            *_, source = q3_overview_source(args, sid)
            print(f"样本 {sid}：数据核对通过；可见词 {source['represented_word_count']}，标点等剩余槽 {source['unmapped_token_count']}。")
        else:
            q3_plot_sample_overview(args, sid)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, AssertionError, KeyError) as exc:
        print(f"生成失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
