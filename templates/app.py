#!/usr/bin/env python3
"""
Chord Detector Web — v3.0 "Wow Factor"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Full-featured chord progression analyser with:
  • Waveform + spectrogram visualisation
  • Chromagram heatmap
  • Roman numeral analysis
  • Famous pattern detection
  • Section detection
  • Chord transition matrix
  • Tension / resolution graph
  • Modulation (key change) detection
  • Scale suggestions for soloing
  • Guitar chord diagrams
  • Circle of fifths journey
  • MIDI / Text / JSON export
"""

import base64, io, json, logging, os, uuid, warnings
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from midiutil import MIDIFile

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
UPLOAD = Path("/tmp/chord-uploads"); UPLOAD.mkdir(exist_ok=True)
ALLOWED = {".mp3",".wav",".flac",".ogg",".m4a",".aac",".wma",".aiff",".opus",".webm"}

# ═══════════════════════════════════════════════════════════════════════════════
# MUSIC THEORY DATA
# ═══════════════════════════════════════════════════════════════════════════════

NOTES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
FLAT  = {"C#":"Db","D#":"Eb","F#":"Gb","G#":"Ab","A#":"Bb"}

def n2f(n):
    """Note name, prefer flats."""
    return FLAT.get(n, n)

# ── Chord templates ──
QUALITIES = {
    "":     [1,0,0,0,1,0,0,1,0,0,0,0],
    "m":    [1,0,0,1,0,0,0,1,0,0,0,0],
    "7":    [1,0,0,0,1,0,0,1,0,0,1,0],
    "m7":   [1,0,0,1,0,0,0,1,0,0,1,0],
    "maj7": [1,0,0,0,1,0,0,1,0,0,0,1],
    "dim":  [1,0,0,1,0,0,1,0,0,0,0,0],
    "aug":  [1,0,0,0,1,0,0,0,1,0,0,0],
    "sus2": [1,0,1,0,0,0,0,1,0,0,0,0],
    "sus4": [1,0,0,0,0,1,0,1,0,0,0,0],
    "add9": [1,0,1,0,1,0,0,1,0,0,0,0],
    "6":    [1,0,0,0,1,0,0,1,0,1,0,0],
    "m6":   [1,0,0,1,0,0,0,1,0,1,0,0],
    "9":    [1,0,1,0,1,0,0,1,0,0,1,0],
    "5":    [1,0,0,0,0,0,0,1,0,0,0,0],
}
Q_PRI = ["","m","5","7","m7","maj7","sus4","sus2","6","m6","dim","aug","add9","9"]

def _build_templates():
    t = {}
    for s, root in enumerate(NOTES):
        for q, base in QUALITIES.items():
            vec = np.roll(base, s).astype(np.float64)
            t[f"{root}{q}"] = vec
            if root in FLAT: t[f"{FLAT[root]}{q}"] = vec.copy()
    return t
TEMPLATES = _build_templates()

CHORD_INTERVALS = {
    "":     [0,4,7],     "m":    [0,3,7],     "7":    [0,4,7,10],
    "m7":   [0,3,7,10],  "maj7": [0,4,7,11],  "dim":  [0,3,6],
    "aug":  [0,4,8],     "sus2": [0,2,7],      "sus4": [0,5,7],
    "add9": [0,2,4,7],   "6":    [0,4,7,9],    "m6":   [0,3,7,9],
    "9":    [0,2,4,7,10],"5":    [0,7],
}

# ── Key profiles ──
MAJ_P = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
MIN_P = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
MAJ_SCALE = [0,2,4,5,7,9,11]
MIN_SCALE = [0,2,3,5,7,8,10]
ROMAN_MAJ = ["I","II","III","IV","V","VI","VII"]
ROMAN_MIN = ["i","ii","iii","iv","v","vi","vii"]

# ── Tension scores (dissonance values for intervals) ──
# Based on Helmholtz consonance rankings
INTERVAL_TENSION = {
    0: 0.0,   # unison
    1: 1.0,   # minor 2nd (most dissonant)
    2: 0.6,   # major 2nd
    3: 0.3,   # minor 3rd
    4: 0.2,   # major 3rd
    5: 0.15,  # perfect 4th
    6: 0.9,   # tritone
    7: 0.1,   # perfect 5th (most consonant non-unison)
    8: 0.2,   # minor 6th
    9: 0.3,   # major 6th
    10: 0.7,  # minor 7th
    11: 0.5,  # major 7th
}

# ── Scale suggestions ──
SCALE_DB = {
    "": [
        {"name": "Major Pentatonic", "intervals": [0,2,4,7,9]},
        {"name": "Ionian (Major)", "intervals": [0,2,4,5,7,9,11]},
        {"name": "Lydian", "intervals": [0,2,4,6,7,9,11]},
        {"name": "Mixolydian", "intervals": [0,2,4,5,7,9,10]},
    ],
    "m": [
        {"name": "Minor Pentatonic", "intervals": [0,3,5,7,10]},
        {"name": "Dorian", "intervals": [0,2,3,5,7,9,10]},
        {"name": "Aeolian (Natural Minor)", "intervals": [0,2,3,5,7,8,10]},
        {"name": "Blues Scale", "intervals": [0,3,5,6,7,10]},
    ],
    "7": [
        {"name": "Mixolydian", "intervals": [0,2,4,5,7,9,10]},
        {"name": "Blues Scale", "intervals": [0,3,5,6,7,10]},
        {"name": "Lydian Dominant", "intervals": [0,2,4,6,7,9,10]},
    ],
    "m7": [
        {"name": "Dorian", "intervals": [0,2,3,5,7,9,10]},
        {"name": "Minor Pentatonic", "intervals": [0,3,5,7,10]},
        {"name": "Aeolian", "intervals": [0,2,3,5,7,8,10]},
    ],
    "maj7": [
        {"name": "Ionian", "intervals": [0,2,4,5,7,9,11]},
        {"name": "Lydian", "intervals": [0,2,4,6,7,9,11]},
    ],
    "dim": [
        {"name": "Whole-Half Diminished", "intervals": [0,2,3,5,6,8,9,11]},
        {"name": "Locrian", "intervals": [0,1,3,5,6,8,10]},
    ],
    "sus4": [
        {"name": "Mixolydian", "intervals": [0,2,4,5,7,9,10]},
    ],
}

# ── Guitar chord voicings (standard tuning: E A D G B E) ──
# Format: [E, A, D, G, B, e] where -1 = muted, 0 = open
GUITAR_CHORDS = {
    "C":    [[None,3,2,0,1,0]],
    "D":    [[None,None,0,2,3,2]],
    "E":    [[0,2,2,1,0,0]],
    "F":    [[1,3,3,2,1,1]],
    "G":    [[3,2,0,0,0,3]],
    "A":    [[None,0,2,2,2,0]],
    "B":    [[None,2,4,4,4,2]],
    "Cm":   [[None,3,5,5,4,3]],
    "Dm":   [[None,None,0,2,3,1]],
    "Em":   [[0,2,2,0,0,0]],
    "Fm":   [[1,3,3,1,1,1]],
    "Gm":   [[3,5,5,3,3,3]],
    "Am":   [[None,0,2,2,1,0]],
    "Bm":   [[None,2,4,4,3,2]],
    "C7":   [[None,3,2,3,1,0]],
    "D7":   [[None,None,0,2,1,2]],
    "E7":   [[0,2,0,1,0,0]],
    "F7":   [[1,3,1,2,1,1]],
    "G7":   [[3,2,0,0,0,1]],
    "A7":   [[None,0,2,0,2,0]],
    "B7":   [[None,2,1,2,0,2]],
    "Cm7":  [[None,3,5,3,4,3]],
    "Dm7":  [[None,None,0,2,1,1]],
    "Em7":  [[0,2,0,0,0,0]],
    "Am7":  [[None,0,2,0,1,0]],
    "Bm7":  [[None,2,4,2,3,2]],
    "Fmaj7":[[None,None,3,2,1,0]],
    "Cmaj7":[[None,3,2,0,0,0]],
    "Gmaj7":[[3,2,0,0,0,2]],
    "Dmaj7":[[None,None,0,2,2,2]],
    "Amaj7":[[None,0,2,1,2,0]],
    "Emaj7":[[0,2,1,1,0,0]],
    "Bmaj7":[[None,2,4,3,4,2]],
    "Bdim": [[None,2,3,4,3,None]],
    "Cdim": [[None,3,4,5,4,None]],
    "Ddim": [[None,None,0,1,0,1]],
    "Edim": [[None,None,2,3,2,3]],
    "Fdim": [[None,None,3,4,3,4]],
    "Gdim": [[3,None,5,3,2,None]],
    "Adim": [[None,0,1,2,1,None]],
    "Csus4":[[None,3,3,0,1,1]],
    "Dsus4":[[None,None,0,2,3,3]],
    "Esus4":[[0,2,2,2,0,0]],
    "Gsus4":[[3,2,0,0,1,3]],
    "Asus4":[[None,0,2,2,3,0]],
    "Bsus4":[[None,2,4,4,5,2]],
    "Fsus4":[[1,3,3,3,1,1]],
    "Csus2":[[None,3,0,0,1,0]],
    "Dsus2":[[None,None,0,2,3,0]],
    "Esus2":[[0,2,4,4,0,0]],
    "Gsus2":[[3,0,0,0,0,3]],
    "Asus2":[[None,0,2,2,0,0]],
    "C5":   [[None,3,5,5,None,None]],
    "D5":   [[None,None,0,2,3,None]],
    "E5":   [[0,2,2,None,None,None]],
    "F5":   [[1,3,3,None,None,None]],
    "G5":   [[3,5,5,None,None,None]],
    "A5":   [[None,0,2,2,None,None]],
    "B5":   [[None,2,4,4,None,None]],
    # Flat-name aliases
    "Db":   [[None,4,3,1,2,1]],
    "Eb":   [[None,None,1,3,4,3]],
    "Gb":   [[2,4,4,3,2,2]],
    "Ab":   [[4,6,6,5,4,4]],
    "Bb":   [[None,1,3,3,3,1]],
    "Dbm":  [[None,4,6,6,5,4]],
    "Ebm":  [[None,None,1,3,4,2]],
    "Gbm":  [[2,4,4,2,2,2]],
    "Abm":  [[4,6,6,4,4,4]],
    "Bbm":  [[None,1,3,3,2,1]],
    "Db7":  [[None,4,3,4,2,1]],
    "Eb7":  [[None,None,1,3,2,3]],
    "Gb7":  [[2,4,2,3,2,2]],
    "Ab7":  [[4,6,4,5,4,4]],
    "Bb7":  [[None,1,3,1,3,1]],
    "Dbm7": [[None,4,6,4,5,4]],
    "Ebm7": [[None,None,1,3,2,2]],
    "Gbm7": [[2,4,2,2,2,2]],
    "Abm7": [[4,6,4,4,4,4]],
    "Bbm7": [[None,1,3,1,2,1]],
}

# ── Famous patterns ──
FAMOUS = {
    "I V vi IV":       ("Axis of Awesome / Pop Punk", "Let It Be, No Woman No Cry, With or Without You"),
    "I IV V I":        ("Classic Blues / 50s Rock", "Twist and Shout, La Bamba, Wild Thing"),
    "I IV V IV":       ("Rock Anthem", "Louie Louie, Born in the USA"),
    "vi IV I V":       ("Minor Start Pop", "Numb, Africa, Save Tonight"),
    "I vi IV V":       ("50s Doo-Wop", "Stand by Me, Every Breath You Take"),
    "ii V I":          ("Jazz ii-V-I Turnaround", "Fly Me to the Moon, Autumn Leaves"),
    "I IV vi V":       ("Modern Pop", "Apologize, Viva la Vida"),
    "i VII VI VII":    ("Andalusian Cadence", "Hit the Road Jack, Sultans of Swing"),
    "I V vi iii IV I IV V": ("Pachelbel's Canon", "Canon in D, Basket Case"),
    "I iii IV V":      ("Emotional Pop", "Creep, Let Her Go"),
    "I bVII IV I":     ("Mixolydian Vamp", "Sweet Child O' Mine, Hey Jude"),
    "i III VII IV":    ("Epic Minor", "Zombie, Boulevard of Broken Dreams"),
    "I V IV V":        ("Classic Rock", "Brown Eyed Girl"),
    "I IV I V":        ("Country / Folk", "Country Roads, Ring of Fire"),
}

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_key(chroma):
    totals = np.mean(chroma, axis=1)
    best_corr, best_key, best_root, best_mode = -2.0, "C", 0, "major"
    for s in range(12):
        rot = np.roll(totals, -s)
        cm = np.corrcoef(rot, MAJ_P)[0,1]
        if cm > best_corr: best_corr, best_key, best_root, best_mode = cm, f"{NOTES[s]} major", s, "major"
        cn = np.corrcoef(rot, MIN_P)[0,1]
        if cn > best_corr: best_corr, best_key, best_root, best_mode = cn, f"{NOTES[s]} minor", s, "minor"
    return best_key, best_root, best_mode, best_corr


def detect_key_at(chroma_slice):
    """Detect key from a chroma slice (for modulation detection)."""
    return detect_key(chroma_slice)


def match_chord(vec, sensitivity=0.55):
    if np.sum(vec) < 0.01: return "N", 0.0
    norm = vec / (np.linalg.norm(vec) + 1e-10)
    best_s, best_c, best_p = -1.0, "N", 999
    for name, tmpl in TEMPLATES.items():
        tn = tmpl / (np.linalg.norm(tmpl) + 1e-10)
        sc = float(np.dot(norm, tn))
        rl = 2 if len(name) > 1 and name[1] in ("#","b") else 1
        q = name[rl:]
        pri = Q_PRI.index(q) if q in Q_PRI else 50
        if sc > best_s + 0.02 or (sc > best_s - 0.005 and pri < best_p):
            if sc >= best_s - 0.005: best_s, best_c, best_p = sc, name, pri
    return ("N", best_s) if best_s < sensitivity else (best_c, best_s)


# ── Chord simplification (reduce complex qualities to basic triads) ──
SIMPLIFY_MAP = {
    "dim": "m", "aug": "", "sus2": "", "sus4": "", "add9": "",
    "6": "", "m6": "m", "9": "7", "5": "",
}

def simplify_chord(name):
    """Reduce a complex chord to its basic quality (major/minor/7/m7/maj7)."""
    if name == "N": return name
    rl = 2 if len(name) > 1 and name[1] in ("#","b") else 1
    root, q = name[:rl], name[rl:]
    if q in SIMPLIFY_MAP:
        return root + SIMPLIFY_MAP[q]
    return name


def merge_short_segments(merged, min_dur):
    """Absorb segments shorter than min_dur into their longest neighbor.
       Repeat until no short non-N segments remain."""
    if min_dur <= 0: return merged
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(merged):
            seg = merged[i]
            dur = seg["end"] - seg["start"]
            if seg["chord"] != "N" and dur < min_dur and len(merged) > 1:
                # pick neighbor to absorb into: prefer the one with same chord,
                # otherwise the longer one
                left  = merged[i-1] if i > 0 else None
                right = merged[i+1] if i < len(merged)-1 else None
                target = None
                if left and left["chord"] == seg["chord"]:
                    target = "left"
                elif right and right["chord"] == seg["chord"]:
                    target = "right"
                elif left and right:
                    ld = left["end"] - left["start"]
                    rd = right["end"] - right["start"]
                    target = "left" if ld >= rd else "right"
                elif left:
                    target = "left"
                elif right:
                    target = "right"
                if target == "left":
                    left["end"] = seg["end"]
                    merged.pop(i)
                    changed = True
                elif target == "right":
                    right["start"] = seg["start"]
                    merged.pop(i)
                    changed = True
                else:
                    i += 1
            else:
                i += 1
        # After absorbing, re-merge consecutive identical chords
        if changed:
            new = []
            for seg in merged:
                if new and new[-1]["chord"] == seg["chord"]:
                    new[-1]["end"] = seg["end"]
                    new[-1]["confidence"] = round((new[-1]["confidence"]+seg["confidence"])/2, 3)
                else:
                    new.append(seg)
            merged = new
    return merged


def _parse_chord(name):
    """Returns (root_index, quality_string) or (None, None)."""
    if name == "N": return None, None
    rl = 2 if len(name) > 1 and name[1] in ("#","b") else 1
    rn, q = name[:rl], name[rl:]
    ri = None
    for i, n in enumerate(NOTES):
        if n == rn: ri = i; break
    if ri is None:
        for s, f in FLAT.items():
            if f == rn: ri = NOTES.index(s); break
    return ri, q


def chord_to_roman(chord, key_root, key_mode):
    ri, q = _parse_chord(chord)
    if ri is None: return "—"
    interval = (ri - key_root) % 12
    scale = MAJ_SCALE if key_mode == "major" else MIN_SCALE
    prefix = ""
    if interval in scale:
        degree = scale.index(interval)
    else:
        closest = min(range(7), key=lambda d: min(abs(scale[d]-interval), 12-abs(scale[d]-interval)))
        degree = closest
        diff = interval - scale[closest]
        prefix = "b" if (diff < 0 or diff > 6) else ("#" if 0 < diff <= 6 else "")
    is_min = q in ("m","m7","m6","dim")
    roman = ROMAN_MIN[degree] if is_min else ROMAN_MAJ[degree]
    suffix = ""
    if q in ("7","9"): suffix = q
    elif q == "m7": suffix = "7"
    elif q == "maj7": suffix = "maj7"
    elif q in ("dim","aug","sus2","sus4"): suffix = q
    return prefix + roman + suffix


def get_chord_notes(chord):
    ri, q = _parse_chord(chord)
    if ri is None: return [], []
    ivs = CHORD_INTERVALS.get(q, [0,4,7])
    pcs = [(ri + iv) % 12 for iv in ivs]
    return pcs, [n2f(NOTES[p]) for p in pcs]


def cof_position(chord):
    """Position on the circle of fifths (0=C, 1=G, 2=D, ... 11=F)."""
    ri, q = _parse_chord(chord)
    if ri is None: return -1
    # CoF order: C G D A E B Gb Db Ab Eb Bb F
    cof = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
    return cof.index(ri) if ri in cof else 0


# ── Transition Matrix ──
def build_transition_matrix(segments):
    """Build an N×N chord transition count matrix."""
    chords_seq = [s["chord"] for s in segments if s["chord"] != "N"]
    unique = sorted(set(chords_seq))
    n = len(unique)
    idx = {c: i for i, c in enumerate(unique)}
    matrix = [[0]*n for _ in range(n)]
    for i in range(len(chords_seq)-1):
        fr, to = chords_seq[i], chords_seq[i+1]
        if fr != to:
            matrix[idx[fr]][idx[to]] += 1
    return {"labels": unique, "matrix": matrix}


# ── Tension / Resolution ──
def chord_tension(chord):
    """Compute a tension score 0-1 for a chord based on its intervals."""
    ri, q = _parse_chord(chord)
    if ri is None: return 0.0
    ivs = CHORD_INTERVALS.get(q, [0,4,7])
    if len(ivs) <= 1: return 0.0
    tensions = []
    for i in range(len(ivs)):
        for j in range(i+1, len(ivs)):
            interval = (ivs[j] - ivs[i]) % 12
            tensions.append(INTERVAL_TENSION.get(interval, 0.5))
    return round(sum(tensions) / len(tensions), 3) if tensions else 0.0


def build_tension_curve(segments):
    """Build a tension-over-time curve."""
    curve = []
    for s in segments:
        t = chord_tension(s["chord"])
        curve.append({
            "time": round((s["start"] + s["end"]) / 2, 2),
            "start": s["start"], "end": s["end"],
            "tension": t, "chord": s["chord"]
        })
    return curve


# ── Modulation Detection ──
def detect_modulations(full_chroma, sr, hop=512, window_sec=8.0):
    """Detect key changes by windowed key analysis."""
    frames_per_window = int(window_sec * sr / hop)
    n_frames = full_chroma.shape[1]
    regions = []
    step = max(1, frames_per_window // 2)

    prev_key = None
    for start in range(0, n_frames - frames_per_window + 1, step):
        end = start + frames_per_window
        chunk = full_chroma[:, start:end]
        key_str, root, mode, _ = detect_key_at(chunk)
        t = librosa.frames_to_time(start, sr=sr, hop_length=hop)
        t_end = librosa.frames_to_time(end, sr=sr, hop_length=hop)

        if key_str != prev_key:
            regions.append({"key": key_str, "start": round(float(t), 2),
                            "end": round(float(t_end), 2)})
            prev_key = key_str
        elif regions:
            regions[-1]["end"] = round(float(t_end), 2)

    # Merge very short regions
    merged = []
    for r in regions:
        if merged and merged[-1]["key"] == r["key"]:
            merged[-1]["end"] = r["end"]
        elif r["end"] - r["start"] >= 3.0:
            merged.append(dict(r))
        elif merged:
            merged[-1]["end"] = r["end"]

    return merged if len(merged) > 1 else []


# ── Scale Suggestions ──
def suggest_scales(chord):
    ri, q = _parse_chord(chord)
    if ri is None: return []
    base_q = q if q in SCALE_DB else ("m" if q in ("m7","m6") else ("7" if q in ("9",) else ("" if q in ("maj7","6","add9","sus2","sus4","5") else "")))
    suggestions = SCALE_DB.get(base_q, SCALE_DB[""])
    result = []
    for s in suggestions:
        notes = [(ri + iv) % 12 for iv in s["intervals"]]
        result.append({
            "name": s["name"],
            "notes": [n2f(NOTES[n]) for n in notes],
        })
    return result


# ── Guitar Voicings ──
def get_guitar_voicing(chord):
    if chord in GUITAR_CHORDS:
        return {"chord": chord, "voicings": GUITAR_CHORDS[chord]}
    return {"chord": chord, "voicings": []}


# ── Famous Patterns ──
def detect_patterns(roman_seq):
    found = []
    for pat, (name, ex) in FAMOUS.items():
        parts = pat.split()
        count = sum(1 for i in range(len(roman_seq)-len(parts)+1) if roman_seq[i:i+len(parts)] == parts)
        if count > 0:
            found.append({"pattern": pat, "name": name, "examples": ex, "occurrences": count})
    return found


# ── Sections ──
def detect_sections(merged):
    chords = [s["chord"] for s in merged if s["chord"] != "N"]
    if len(chords) < 8: return []
    best_pat, best_cnt = None, 0
    for pl in range(4, min(9, len(chords)//2+1)):
        pats = Counter(tuple(chords[i:i+pl]) for i in range(len(chords)-pl+1))
        for p, c in pats.most_common(3):
            if c >= 2 and c > best_cnt: best_cnt, best_pat = c, p
    if not best_pat: return []
    plen = len(best_pat)
    non_n = [s for s in merged if s["chord"] != "N"]
    lmap, lc = {}, 0
    LABELS = ["A (Verse)","B (Chorus)","C (Bridge)","D","E"]
    secs = []
    i = 0
    while i <= len(non_n) - plen:
        w = tuple(s["chord"] for s in non_n[i:i+plen])
        if w not in lmap:
            lmap[w] = LABELS[lc] if lc < len(LABELS) else f"Section {lc+1}"
            lc += 1
        secs.append({"label": lmap[w], "start": non_n[i]["start"],
                      "end": non_n[i+plen-1]["end"], "chords": list(w)})
        i += plen
    return secs


# ── Visualisation data ──
def extract_waveform(y, n=1500):
    hop = max(1, len(y)//n)
    frames = librosa.util.frame(y, frame_length=hop, hop_length=hop)
    peaks = np.max(np.abs(frames), axis=0)
    r = []
    for p in peaks: r += [round(float(p),4), round(float(-p),4)]
    return r[:n*2]


def extract_chromagram(chroma, cols=250):
    if chroma.shape[1] > cols:
        f = chroma.shape[1]//cols
        chroma = chroma[:, ::f][:, :cols]
    mx = np.max(chroma)
    if mx > 0: chroma /= mx
    return [[round(float(v),3) for v in row] for row in chroma]


def extract_spectrogram(y, sr, cols=300):
    """Compute mel spectrogram for visualisation."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=1024)
    S_db = librosa.power_to_db(S, ref=np.max)
    # Normalise to 0-1
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-10)
    if S_db.shape[1] > cols:
        f = S_db.shape[1]//cols
        S_db = S_db[:, ::f][:, :cols]
    return [[round(float(v),3) for v in row] for row in S_db]


def generate_midi(segments, tempo, key_str):
    midi = MIDIFile(1)
    midi.addTempo(0, 0, tempo or 120)
    midi.addTrackName(0, 0, f"Chords - {key_str}")
    bpm = tempo or 120
    for seg in segments:
        if seg["chord"] == "N": continue
        notes, _ = get_chord_notes(seg["chord"])
        dur = seg["end"] - seg["start"]
        for npc in notes:
            midi.addNote(0, 0, 60+npc, seg["start"]*(bpm/60), dur*(bpm/60), 80)
    buf = io.BytesIO(); midi.writeFile(buf); buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def analyse(filepath, sensitivity=0.55, min_duration=0.0, simplify=False, hpss_margin=2.0):
    y, sr = librosa.load(filepath, sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # Harmonic-percussive separation with adjustable margin
    # Higher margin = more aggressive vocal removal (keeps only strong harmonics)
    y_harm, _ = librosa.effects.hpss(y, margin=hpss_margin)

    tempo, beat_frames = librosa.beat.beat_track(y=y_harm, sr=sr)
    if isinstance(tempo, np.ndarray): tempo = float(tempo[0])

    chroma_cqt = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=512, n_chroma=12)

    # Smooth the chromagram to reduce vocal/noise artifacts
    if min_duration > 0:
        smooth_frames = max(1, int(min_duration * sr / 512 / 4))
        from scipy.ndimage import uniform_filter1d
        chroma_cqt = uniform_filter1d(chroma_cqt, size=smooth_frames, axis=1)

    if len(beat_frames) > 1:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        chroma_sync = librosa.util.sync(chroma_cqt, beat_frames, aggregate=np.median)
    else:
        hop = 512; win = int(0.5*sr/hop)
        indices = np.arange(0, chroma_cqt.shape[1], win)
        beat_frames, beat_times = indices, librosa.frames_to_time(indices, sr=sr, hop_length=hop)
        chroma_sync = librosa.util.sync(chroma_cqt, indices, aggregate=np.median)
        if tempo == 0: tempo = 120.0

    full_chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    key_str, key_root, key_mode, key_conf = detect_key(full_chroma)

    # Chord matching
    segments = []
    for i in range(chroma_sync.shape[1]):
        ch, conf = match_chord(chroma_sync[:, i], sensitivity)
        st = float(beat_times[i]) if i < len(beat_times) else 0.0
        en = float(beat_times[i+1]) if i+1 < len(beat_times) else duration
        segments.append({"chord":ch,"start":round(st,3),"end":round(en,3),"confidence":round(conf,3)})

    # Merge consecutive
    merged = []
    for seg in segments:
        if merged and merged[-1]["chord"] == seg["chord"]:
            merged[-1]["end"] = seg["end"]
            merged[-1]["confidence"] = round((merged[-1]["confidence"]+seg["confidence"])/2, 3)
        else:
            merged.append(dict(seg))

    # Use flats
    for seg in merged:
        c = seg["chord"]
        if len(c) > 1 and c[1] == "#":
            r, q = c[:2], c[2:]
            if r in FLAT: seg["chord"] = FLAT[r] + q

    merged = [s for s in merged if not (s["chord"]=="N" and (s["end"]-s["start"])<0.3)]

    # Simplify chords (reduce complex qualities to basic triads)
    if simplify:
        for seg in merged:
            seg["chord"] = simplify_chord(seg["chord"])
        # Re-merge after simplification (e.g. Csus4 + C → C + C → C)
        new = []
        for seg in merged:
            if new and new[-1]["chord"] == seg["chord"]:
                new[-1]["end"] = seg["end"]
                new[-1]["confidence"] = round((new[-1]["confidence"]+seg["confidence"])/2, 3)
            else:
                new.append(seg)
        merged = new

    # Absorb short segments into neighbors
    merged = merge_short_segments(merged, min_duration)

    # Also absorb short N gaps when min_duration is set
    if min_duration > 0:
        merged = [s for s in merged if not (s["chord"]=="N" and (s["end"]-s["start"]) < min_duration)]
        # Re-merge again after removing N gaps
        new = []
        for seg in merged:
            if new and new[-1]["chord"] == seg["chord"]:
                new[-1]["end"] = seg["end"]
                new[-1]["confidence"] = round((new[-1]["confidence"]+seg["confidence"])/2, 3)
            else:
                new.append(seg)
        merged = new

    # Enrich segments
    for seg in merged:
        seg["roman"] = chord_to_roman(seg["chord"], key_root, key_mode)
        pcs, nn = get_chord_notes(seg["chord"])
        seg["notes"] = pcs
        seg["note_names"] = nn
        seg["tension"] = chord_tension(seg["chord"])
        seg["cof_pos"] = cof_position(seg["chord"])
        seg["scales"] = suggest_scales(seg["chord"])
        seg["guitar"] = get_guitar_voicing(seg["chord"])

    # Progression
    prog = [s["chord"] for s in merged if s["chord"] != "N"]
    deduped = []
    for c in prog:
        if not deduped or deduped[-1] != c: deduped.append(c)
    roman_seq = [chord_to_roman(c, key_root, key_mode) for c in deduped]

    # Stats
    stats = {}
    for s in merged:
        if s["chord"] != "N": stats[s["chord"]] = stats.get(s["chord"],0) + (s["end"]-s["start"])
    td = sum(stats.values()) or 1.0
    chord_stats = [{"chord":c,"duration":round(d,1),"percent":round(d/td*100,1)}
                    for c,d in sorted(stats.items(), key=lambda x:-x[1])]

    return {
        "key": key_str, "key_confidence": round(key_conf,3),
        "tempo": round(tempo), "duration": round(duration,2),
        "segments": merged,
        "progression": deduped,
        "roman_progression": roman_seq,
        "patterns": detect_patterns(roman_seq),
        "sections": detect_sections(merged),
        "chord_stats": chord_stats,
        "unique_chords": sorted(set(deduped)),
        "transition_matrix": build_transition_matrix(merged),
        "tension_curve": build_tension_curve(merged),
        "modulations": detect_modulations(full_chroma, sr),
        "waveform": extract_waveform(y),
        "chromagram": extract_chromagram(full_chroma),
        "spectrogram": extract_spectrogram(y, sr),
        "beat_positions": [round(float(bt),3) for bt in beat_times] if len(beat_times)>1 else [],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    log.info("Serving index page")
    return render_template("index.html")

@app.route("/health")
def health(): return "ok", 200

@app.errorhandler(500)
def handle_500(e):
    log.exception("Internal server error")
    return jsonify({"error": str(e)}), 500

@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    if "file" not in request.files: return jsonify({"error":"No file"}), 400
    f = request.files["file"]
    if not f.filename: return jsonify({"error":"No file"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED: return jsonify({"error":f"Unsupported: {ext}"}), 400
    sens = max(0.1, min(1.0, float(request.form.get("sensitivity",0.55))))
    min_dur = max(0.0, min(8.0, float(request.form.get("min_duration", 0.0))))
    simplify = request.form.get("simplify", "false").lower() in ("true","1","yes","on")
    # HPSS margin: higher = more aggressive vocal/noise removal
    # 2.0 = default, up to 8.0 for heavy vocal tracks
    hpss_margin = max(1.0, min(8.0, float(request.form.get("hpss_margin", 2.0))))
    uid = uuid.uuid4().hex[:12]
    tmp = UPLOAD/f"{uid}{ext}"; f.save(str(tmp))
    log.info(f"Analysing {f.filename} (sens={sens}, min_dur={min_dur}, simplify={simplify}, hpss={hpss_margin})")
    try:
        result = analyse(str(tmp), sens, min_dur, simplify, hpss_margin)
        result["filename"] = f.filename
        with open(str(tmp),"rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        mimes = {".mp3":"audio/mpeg",".wav":"audio/wav",".flac":"audio/flac",".ogg":"audio/ogg",".m4a":"audio/mp4",".aac":"audio/aac",".webm":"audio/webm",".aiff":"audio/aiff",".opus":"audio/opus"}
        result["audio_data"] = f"data:{mimes.get(ext,'audio/mpeg')};base64,{b64}"
        return jsonify(result)
    except Exception as e:
        log.exception(f"Analysis failed for {f.filename}")
        return jsonify({"error":f"Analysis failed: {e}"}), 500
    finally:
        tmp.unlink(missing_ok=True)

@app.route("/api/export/midi", methods=["POST"])
def export_midi():
    d = request.json
    if not d: return jsonify({"error":"No data"}), 400
    buf = generate_midi(d.get("segments",[]), d.get("tempo",120), d.get("key","C"))
    return send_file(buf, mimetype="audio/midi", as_attachment=True, download_name="chords.mid")

log.info("Chord Detector v3.0 loaded — ready to serve")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",5000)), debug=False)
