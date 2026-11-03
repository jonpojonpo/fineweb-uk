"""The five pilot decisions. Each is one question with a closed answer set."""
# FineWeb-Edu's published scoring prompt (Penedo et al. 2024, arXiv:2406.17557, appendix).
RUBRIC = ("Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching from primary school to grade school levels using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:\n"
"- Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.\n"
"- Add another point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing style.\n"
"- Award a third point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a basic tutorial that is suitable for learning but has notable limitations like treating concepts that are too complex for grade school students.\n"
"- Grant a fourth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren't too advanced for grade school students. The content is coherent, focused, and valuable for structured learning.\n"
"- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or grade school. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.\n"
"Give the total score; a page meeting no criterion scores 0.")
Q = {
 "quality": {"prompt": RUBRIC,
   "options": {str(i): str(i) for i in range(6)}, "ordinal": True},
 "uk_relevance": {"prompt": "How does this text relate to the United Kingdom?",
   "options": {"uk_specific": "specifically about the UK, its institutions, places, law or culture",
               "uk_compatible": "not about the UK, but written for or consistent with a British reader",
               "non_uk": "about or written for somewhere else, and not useful to a British reader"}},
 "genre": {"prompt": "What is the genre of this text?",
   "options": {k: k for k in ["news","legal","government","forum","commercial","academic","creative","other"]}},
 "spam_or_boilerplate": {"prompt": "Is this text mostly spam, SEO filler, or site boilerplate?", "options": {"yes":"yes","no":"no"}},
 "pii_risk": {"prompt": "Does this text expose personal data about a private individual, such as a home address, phone number or private email?", "options": {"yes":"yes","no":"no"}},
}
LETTERS = "ABCDEFGH"

def render(q, doc, reverse=False, max_chars=6000):
    """Returns (prompt, {answer_token: option}). Quality answers are digits; the rest are letters."""
    keys = list(q["options"])
    if q.get("ordinal"):
        ans = {k: k for k in keys}
        lines = ""
        tail = "Answer with a single digit, 0 to 5."
    else:
        if reverse: keys = keys[::-1]
        ans = {LETTERS[i]: k for i, k in enumerate(keys)}
        lines = "\n".join(f"{l}. {k}: {q['options'][k]}" if q['options'][k] != k else f"{l}. {k}" for l, k in ans.items()) + "\n"
        tail = "Answer with a single letter."
    return f"<document>\n{doc[:max_chars]}\n</document>\n\n{q['prompt']}\n{lines}{tail}", ans

# Croydon: how much a document tells a model about the UK. FineWeb-Edu-style 0-5 scale; the score kept is the
# probability-weighted average, so it ranks finely (e.g. 3.47) while staying comparable to the Edu axis.
CROYDON_RUBRIC = """Rate how much this page would teach a language model about the United Kingdom, from 0 to 5:
0: It is about or written for another country, so a British reader would need to adapt it (for example US tax advice or a local US news story).
1: Neutral or global content with no particular UK framing (for example maths, physics, or the history of ancient Egypt).
2: Suited to a British reader, through British spelling, units, prices or context, but not about the UK itself.
3: Partly about the UK, or mentions UK places, institutions, laws or people along the way.
4: Mainly about the UK: its law, government, public services, places, history, culture, sport or current affairs.
5: Rich, specific and accurate UK knowledge that a British model should know, explained in depth (for example how council tax works, the history of the NHS, or a detailed account of a British historical event).
Judge the substance, not the website's address."""
CROYDON = {"prompt": CROYDON_RUBRIC, "options": {str(i): str(i) for i in range(6)}, "ordinal": True}

# Hand-check question sets for label_tool.py. "croydon" asks for a person's own Croydon score and whether to keep the page.
CHECK_SETS = {
    "croydon": {
        "croydon": {"prompt": "How much would this page teach a model about the UK? (0 to 5, same scale as Croydon)", "options": {str(i): str(i) for i in range(6)}, "ordinal": True},
        "keep": {"prompt": "Would you want this page in a UK model's training data?", "options": {"yes": "yes", "maybe": "maybe", "no": "no"}},
    },
}
