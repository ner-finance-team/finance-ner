# -*- coding: utf-8 -*-
"""
팀원이 보낸 결과 압축 파일(results_{이름}_{시각}.zip)을 내 reports/results 로 합친다. 비교·그림은 R0 노트북으로.

검사
- 채점 기준이 같은지: dev 10,000문장 · test 10,000문장으로 채점됐는지(점수 파일의 n_sentences) 확인. 다르면 건너뜀.
- 이름 충돌: 같은 실험 이름이 이미 있고 실행자·완료 시각이 다르면 이름 끝에 __{실행자}를 붙여 따로 저장(덮어쓰지 않음).
- 같은 결과(같은 실행자·같은 완료 시각)를 다시 받으면 건너뜀.

사용 (프로젝트 폴더에서):
  python src/merge_results.py 받은폴더/results_hong_1022_1530.zip 받은폴더/results_kim_*.zip
  python src/merge_results.py 받은폴더/            # 폴더 안의 zip 전부
"""
import argparse, glob, json, os, shutil, sys, tempfile, zipfile

OUT = "reports/results"
NEED_N = {"dev": 10000, "test": 10000}
# 팀원이 노트북 예시 이름을 그대로 쓴 경우 실제 이름으로 (10/22 결과보고 기준)
MEMBER_MAP = {"hong": "kim_byeongseok", "jake": "park_gyutae", "jin": "jin_taehyeon"}

def jload(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

def jdump(o, p):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f: json.dump(o, f, ensure_ascii=False, indent=1)

def find_root(d):
    """압축 안에서 {exp}.json 과 scores/ 가 있는 폴더를 찾는다."""
    for root, dirs, files in os.walk(d):
        if "scores" in dirs: return root
    return d

def merge_zip(zp, out, log):
    tmp = tempfile.mkdtemp()
    with zipfile.ZipFile(zp) as z: z.extractall(tmp)
    root = find_root(tmp)
    names = sorted({os.path.basename(p)[:-len("_test.json")] for p in glob.glob(os.path.join(root, "scores", "*_test.json"))})
    runs_new = []; added = 0
    for name in names:
        sc = {sp: os.path.join(root, "scores", f"{name}_{sp}.json") for sp in ("dev", "test")}
        if not all(os.path.exists(p) for p in sc.values()):
            log.append(f"  건너뜀 {name}: dev/test 점수 파일 중 하나가 없음"); continue
        S = {sp: jload(p) for sp, p in sc.items()}
        bad = [sp for sp in ("dev", "test") if S[sp].get("n_sentences") != NEED_N[sp]]
        if bad:
            log.append(f"  건너뜀 {name}: {bad} 문장 수가 기준(1만)과 다름 → 다른 평가 데이터로 채점한 것"); continue
        exp_json = os.path.join(root, f"{name}.json")
        R = jload(exp_json) if os.path.exists(exp_json) else None
        member = ((R or {}).get("config") or (R or {}).get("meta") or {}).get("member") or S["test"].get("member") or "unknown"
        member = MEMBER_MAP.get(member, member)
        stamp = (R or {}).get("finished_at") or (R or {}).get("recorded_at") or S["test"].get("scored_at")
        target = name
        cur = os.path.join(out, f"{name}.json"); cur_sc = os.path.join(out, "scores", f"{name}_test.json")
        if os.path.exists(cur) or os.path.exists(cur_sc):
            old = jload(cur) if os.path.exists(cur) else {}
            old_member = MEMBER_MAP.get(*(2*[(old.get("config") or old.get("meta") or {}).get("member") or jload(cur_sc).get("member")]))
            old_stamp = old.get("finished_at") or old.get("recorded_at") or jload(cur_sc).get("scored_at")
            if old_member == member and old_stamp == stamp:
                log.append(f"  이미 있음 {name} ({member})"); continue
            target = f"{name}__{member}"
            log.append(f"  이름 충돌 {name}: 기존({old_member}) 보존, 새 결과는 {target} 로 저장")
        # 저장 (이름이 바뀌면 파일 안의 이름도 바꿈)
        for sp in ("dev", "test"):
            S[sp]["name"] = target; jdump(S[sp], os.path.join(out, "scores", f"{target}_{sp}.json"))
            for sub in ("preds", "errors"):
                src = os.path.join(root, sub, f"{name}_{sp}.jsonl")
                if os.path.exists(src):
                    os.makedirs(os.path.join(out, sub), exist_ok=True)
                    shutil.copy2(src, os.path.join(out, sub, f"{target}_{sp}.jsonl"))
        if R:
            R["exp"] = target; jdump(R, os.path.join(out, f"{target}.json"))
            cfg = R.get("config") or R.get("meta") or {}     # 우리 노트북은 config, 팀원 자체 코드는 meta
            runs_new.append({"exp": target, "stage": cfg.get("stage", ""), "member": member,
                             "model": cfg.get("model"), "n_train": R.get("n_train") or cfg.get("train_size"),
                             "dev_f1": S["dev"]["metrics"]["ALL"]["f1"], "test_f1": S["test"]["metrics"]["ALL"]["f1"],
                             "test_macro_f1": S["test"]["metrics"]["MACRO_F1"], "train_sec": R.get("train_sec") or R.get("train_runtime"),
                             "gpu": R.get("gpu"), "finished_at": stamp, "merged_from": os.path.basename(zp)})
        added += 1
        log.append(f"  추가 {target} ({member}) test F1 {S['test']['metrics']['ALL']['f1']}")
    if runs_new:
        with open(os.path.join(out, "runs.jsonl"), "a", encoding="utf-8") as f:
            for r in runs_new: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    shutil.rmtree(tmp, ignore_errors=True)
    return added

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("paths", nargs="+"); ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    zips = []
    for p in a.paths:
        zips += sorted(glob.glob(os.path.join(p, "*.zip"))) if os.path.isdir(p) else glob.glob(p)
    total = 0
    for zp in zips:
        log = []; n = merge_zip(zp, a.out, log); total += n
        print(f"[{os.path.basename(zp)}] {n}개 추가"); print("\n".join(log))
    print(f"합계 {total}개 실험 추가 → {a.out}. 비교는 R0_결과종합 노트북에서.")

if __name__ == "__main__":
    main()
