#!/usr/bin/env python3
"""Generate Needle-in-a-Haystack (NIAH) test datasets for TurboQuant evaluation.

Produces two JSON files:
  - niah_single_needle.json  — one needle per test case
  - niah_multi_needle.json   — multiple needles per test case

Each file contains test cases across context lengths:
  2k, 8k, 24k, 32k, 64k, 128k, 256k tokens.
"""

import json
import os
import random
import textwrap

CHARS_PER_TOKEN = 4  # conservative estimate for English text

CONTEXT_LENGTHS = [2_048, 8_192, 24_576, 32_768, 65_536, 131_072, 262_144]

DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]

# ---------------------------------------------------------------------------
# Haystack filler — diverse, boring paragraphs on unrelated topics
# ---------------------------------------------------------------------------
HAYSTACK_PARAGRAPHS = [
    "The history of paper manufacturing dates back to ancient China, where the process was first developed during the Han Dynasty around 105 AD. Cai Lun, a court official, is traditionally credited with improving the technique by using tree bark, hemp, old rags, and fishnets. The resulting material was lighter and cheaper than bamboo strips or silk, making it ideal for record keeping. Paper production gradually spread westward along the Silk Road, reaching the Islamic world by the 8th century and Europe by the 11th century. Each region adapted the process using locally available fibers, leading to distinct textures and qualities.",
    "Modern weather forecasting relies on a combination of satellite imagery, ground-based observation stations, and numerical weather prediction models. These models divide the atmosphere into a three-dimensional grid and solve fluid dynamics equations at each grid point. The resolution of the grid determines the level of detail in the forecast: global models typically use grid cells around 10 to 25 kilometers wide, while regional models can go below 3 kilometers. Despite these advances, forecasts beyond 10 days remain inherently unreliable due to the chaotic nature of atmospheric dynamics.",
    "The cultivation of coffee beans involves a delicate balance of altitude, rainfall, and temperature. Arabica coffee, which accounts for roughly 60 percent of global production, thrives at elevations between 1,200 and 2,200 meters with annual rainfall of 1,500 to 2,500 millimeters. The beans are typically harvested once a year, either by hand-picking ripe cherries or through strip harvesting. After harvesting, the beans undergo processing—either washed, natural, or honey—each method imparting different flavor profiles to the final cup.",
    "Concrete is the most widely used construction material on Earth, with roughly 30 billion tons produced annually. Its primary components—cement, water, sand, and gravel—are mixed in precise ratios to achieve desired strength characteristics. The hydration reaction between cement and water generates calcium silicate hydrate, which binds the aggregate particles together. Modern concrete technology includes admixtures that can accelerate or retard setting times, improve workability, or enhance durability against freeze-thaw cycles and chemical attack.",
    "The postal system in the United Kingdom underwent a major transformation in 1840 with the introduction of the Penny Black, the world's first adhesive postage stamp. Before this reform, postage was typically paid by the recipient and calculated based on the number of sheets and distance traveled. Sir Rowland Hill proposed a flat-rate prepaid system, arguing it would simplify administration and increase mail volume. The reform proved enormously successful: letter volume in the UK doubled within two years and continued to grow throughout the Victorian era.",
    "Photosynthesis converts light energy into chemical energy through two main stages: the light-dependent reactions and the Calvin cycle. In the light-dependent reactions, which occur in the thylakoid membranes, water molecules are split to release oxygen, and the energy captured by chlorophyll is used to produce ATP and NADPH. The Calvin cycle, occurring in the stroma, uses these energy carriers to fix carbon dioxide into three-carbon sugars. A single leaf can fix approximately 5 micromoles of carbon dioxide per square meter per second under optimal conditions.",
    "The Trans-Siberian Railway, completed in 1916, spans 9,289 kilometers from Moscow to Vladivostok, making it the longest railway line in the world. Construction began in 1891 under Tsar Alexander III and involved overcoming enormous logistical challenges, including crossing permafrost, bridging major rivers, and tunneling through mountain ranges. Today the journey takes approximately six days and passes through eight time zones. The railway remains a critical transport corridor for both passengers and freight in the Russian Federation.",
    "Typography, the art of arranging type, has evolved significantly since Gutenberg's movable type printing press in the 1440s. Early typefaces were designed to mimic handwriting, but over centuries, distinct classifications emerged: serif, sans-serif, slab-serif, script, and display. The choice of typeface affects readability, tone, and perceived credibility. Studies have shown that serif fonts like Times New Roman are often preferred for printed body text, while sans-serif fonts like Helvetica are favored for digital screens due to their cleaner rendering at lower resolutions.",
    "Glaciers cover approximately 10 percent of the Earth's land surface and contain about 69 percent of the world's fresh water. They form in regions where annual snowfall exceeds annual snowmelt, causing layers of snow to compress into dense ice over decades. Glaciers move under their own weight through a combination of internal deformation and basal sliding. The study of glacial movement has become increasingly important as climate change accelerates ice loss, contributing to sea-level rise and altering freshwater availability in downstream regions.",
    "The domestication of wheat began approximately 10,000 years ago in the Fertile Crescent, a region spanning parts of modern-day Iraq, Syria, Lebanon, and Turkey. Early farmers selected for traits like larger seeds, non-shattering seed heads, and shorter growing periods. Over millennia, cross-breeding between wild einkorn, emmer, and goat grass produced bread wheat, which now accounts for roughly 95 percent of global wheat production. Wheat provides about 20 percent of the calories consumed by humans worldwide and is a staple crop on every continent except Antarctica.",
    "Marine navigation before the invention of the chronometer relied on a combination of dead reckoning, celestial observations, and coastal piloting. Latitude could be determined relatively accurately by measuring the angle of the sun or Polaris above the horizon, but longitude remained a persistent problem. The issue was so critical that in 1714 the British Parliament offered a prize of 20,000 pounds for a practical method of determining longitude at sea. John Harrison eventually solved the problem with his series of marine chronometers, the most famous being H4, completed in 1761.",
    "The human circulatory system pumps approximately 7,500 liters of blood through the body each day. The heart, a muscular organ roughly the size of a fist, beats about 100,000 times daily, propelling blood through a network of arteries, veins, and capillaries that, if laid end to end, would stretch over 96,000 kilometers. Red blood cells, numbering about 25 trillion in an adult, carry oxygen from the lungs to tissues and return carbon dioxide for exhalation. Each red blood cell completes a full circuit of the body in roughly 20 seconds.",
    "Volcanic eruptions are classified using the Volcanic Explosivity Index, which ranges from 0 for gentle effusive eruptions to 8 for mega-colossal events. The index considers the volume of ejecta, the height of the eruption column, and the duration of the event. The most recent VEI-8 eruption was the Oruanui eruption in New Zealand approximately 26,500 years ago. More recent large eruptions include Tambora in 1815, a VEI-7 event that caused the 'Year Without a Summer' in 1816 due to the global cooling effect of sulfur aerosols injected into the stratosphere.",
    "The invention of the transistor at Bell Labs in 1947 by John Bardeen, Walter Brattain, and William Shockley fundamentally changed electronics. Transistors replaced vacuum tubes, which were bulky, fragile, and power-hungry. The first transistor was a point-contact device made of germanium, but silicon quickly became the preferred semiconductor material due to its abundance and thermal stability. Modern integrated circuits can contain billions of transistors, each measuring just a few nanometers across, enabling the computational power found in today's smartphones and data centers.",
    "Beekeeping, or apiculture, has been practiced for at least 9,000 years, as evidenced by rock paintings in Spain and beeswax residues found in ancient pottery. A typical honey bee colony consists of one queen, several hundred drones, and 20,000 to 60,000 worker bees. Workers forage for nectar and pollen within a radius of about 5 kilometers from the hive, visiting between 50 and 1,000 flowers per trip. To produce one kilogram of honey, bees must collectively visit approximately 4 million flowers and fly a cumulative distance equivalent to four times around the Earth.",
    "The Dewey Decimal Classification system, created by Melvil Dewey in 1876, organizes library materials into ten main classes, each divided into ten divisions and further into ten sections. Despite its age, the system remains in use in over 200,000 libraries across 135 countries. The system assigns a numerical code to each subject area: 000 for general works, 100 for philosophy, 200 for religion, 300 for social sciences, 400 for language, 500 for science, 600 for technology, 700 for arts, 800 for literature, and 900 for history and geography.",
    "Soil formation is a slow process influenced by five key factors: parent material, climate, organisms, topography, and time. It typically takes 200 to 1,000 years to form one centimeter of topsoil, depending on conditions. Soil is organized into horizontal layers called horizons: the O horizon contains organic matter, the A horizon is the topsoil where most biological activity occurs, the B horizon accumulates minerals leached from above, and the C horizon consists of partially weathered parent material. Healthy topsoil contains roughly 45 percent minerals, 25 percent water, 25 percent air, and 5 percent organic matter.",
    "The metric system was first adopted in France in 1795 during the French Revolution as part of an effort to standardize measurements across the country. The meter was originally defined as one ten-millionth of the distance from the equator to the North Pole along a meridian passing through Paris. Today, the meter is defined in terms of the speed of light: it is the distance light travels in a vacuum in 1/299,792,458 of a second. The International System of Units now comprises seven base units: meter, kilogram, second, ampere, kelvin, mole, and candela.",
    "The production of olive oil involves harvesting olives, usually between October and January in the Northern Hemisphere, and pressing them to extract the oil. Extra virgin olive oil, the highest grade, must be produced entirely by mechanical means without the use of solvents or excessive heat, and must have a free acidity of no more than 0.8 percent. A mature olive tree produces between 15 and 40 kilograms of olives per year, yielding approximately 3 to 8 liters of oil. The Mediterranean region accounts for roughly 95 percent of global olive oil production.",
    "The study of linguistics encompasses phonetics, phonology, morphology, syntax, semantics, and pragmatics. Human languages exhibit remarkable structural diversity: some languages like Mandarin Chinese use tonal distinctions to differentiate meaning, while others like Turkish use extensive agglutination to build complex words from chains of suffixes. Despite this diversity, linguists have identified certain universal tendencies, such as the preference for subject-verb-object or subject-object-verb word orders, which together account for roughly 90 percent of the world's languages.",
    "Railroad gauge, the distance between the inner sides of the two rails, varies significantly around the world. The most common gauge is 1,435 millimeters, known as standard gauge, used by about 60 percent of the world's railways. Russia and the former Soviet states use 1,520-millimeter broad gauge, while much of the Indian subcontinent uses 1,676-millimeter gauge. Narrow gauge railways, with gauges less than 1,435 millimeters, were historically popular in mountainous regions where tight curves and steep gradients made standard gauge impractical. Gauge differences at international borders require solutions such as variable-gauge wheelsets or bogie exchange systems.",
    "The concept of insurance dates back to ancient Babylon, where merchants receiving loans would pay an additional sum to guarantee the loan would be cancelled if the shipment was lost. Modern marine insurance originated in the coffeehouses of London in the late 17th century, where merchants and shipowners would gather to share risks. Lloyd's of London, perhaps the most famous insurance market in the world, began as Edward Lloyd's coffee house around 1688. Today, the global insurance industry generates annual premiums exceeding 6 trillion dollars and employs millions of people worldwide.",
    "Cartography, the science and art of map-making, has undergone dramatic changes with the advent of satellite imagery and geographic information systems. Traditional map projections, such as the Mercator projection developed in 1569, distort the size of landmasses, particularly near the poles. The Mercator projection makes Greenland appear roughly the same size as Africa, despite Africa being approximately 14 times larger. Modern digital mapping tools allow users to switch between projections and view the Earth in three dimensions, but the fundamental challenge of representing a spherical surface on a flat plane remains.",
    "The water cycle describes the continuous movement of water through the Earth's systems. Evaporation from oceans, lakes, and rivers accounts for roughly 90 percent of atmospheric moisture, with the remaining 10 percent coming from plant transpiration. Once in the atmosphere, water vapor condenses to form clouds and eventually precipitates as rain, snow, sleet, or hail. Globally, approximately 505,000 cubic kilometers of water evaporate each year, and about 398,000 cubic kilometers of that falls back into the oceans. The average residence time of a water molecule in the atmosphere is about 9 days.",
    "The periodic table of elements, first published by Dmitri Mendeleev in 1869, organizes the 118 known elements by atomic number and recurring chemical properties. Mendeleev's original table contained gaps that he boldly predicted would be filled by undiscovered elements, specifying their expected properties. The subsequent discovery of gallium, scandium, and germanium confirmed his predictions and cemented the periodic table as a foundational tool in chemistry. The most recent additions to the table, elements 113 through 118, were officially named in 2016.",
    "The Amazon River basin contains approximately 20 percent of the world's fresh water flowing into the oceans. The river itself stretches about 6,400 kilometers from its source in the Peruvian Andes to its mouth on the Atlantic coast of Brazil. During the rainy season, the river can be more than 40 kilometers wide in places, and its discharge at the mouth averages about 209,000 cubic meters per second. The basin's tropical rainforest covers roughly 5.5 million square kilometers and is home to an estimated 10 percent of all species on Earth.",
    "The development of refrigeration technology in the 19th century transformed food storage, transportation, and diet. Before mechanical refrigeration, ice harvested from frozen lakes and rivers was the primary means of preserving perishable foods. The first practical refrigeration machine was built by James Harrison in Australia in 1856. The introduction of refrigerated railcars in the 1870s allowed meat and produce to be shipped across continents, fundamentally changing agricultural economics and enabling year-round access to seasonal foods in distant markets.",
    "Coral reefs occupy less than 0.1 percent of the ocean floor but support approximately 25 percent of all marine species. They are built primarily by colonies of tiny animals called coral polyps, which secrete calcium carbonate to form hard skeletons. A healthy reef grows at a rate of 1 to 3 centimeters per year. The Great Barrier Reef, the largest coral reef system in the world, stretches over 2,300 kilometers along the northeast coast of Australia and is visible from space. Rising ocean temperatures have caused increasingly frequent and severe coral bleaching events.",
    "The practice of crop rotation, alternating different crops in the same field across seasons, has been used since ancient Roman times to maintain soil fertility and reduce pest pressure. A typical four-year rotation might include wheat, turnips, barley, and clover. Legumes like clover and peas fix atmospheric nitrogen through symbiotic bacteria in their root nodules, enriching the soil for subsequent nitrogen-demanding crops. Modern precision agriculture uses soil sensors, GPS-guided equipment, and satellite imagery to optimize rotation patterns and input application at a sub-field scale.",
    "The construction of medieval cathedrals represented some of the most ambitious engineering projects of their era, often taking decades or even centuries to complete. Notre-Dame de Paris, begun in 1163, was not substantially completed until 1260. The builders used flying buttresses—external arched supports—to transfer the weight of the vaulted ceilings to the ground, allowing for taller walls with larger windows. Stained glass windows served both decorative and didactic purposes, illustrating biblical narratives and the lives of saints for a largely illiterate population.",
    "The field of astronomy has been revolutionized by space-based telescopes that observe the universe across multiple wavelengths of the electromagnetic spectrum. The Hubble Space Telescope, launched in 1990, has captured images of galaxies billions of light-years away and helped refine estimates of the age of the universe to approximately 13.8 billion years. Its successor, the James Webb Space Telescope launched in 2021, operates primarily in the infrared and can observe even more distant and older celestial objects, peering back to within a few hundred million years of the Big Bang.",
]

# ---------------------------------------------------------------------------
# Needles — distinctive facts that are easy to verify
# ---------------------------------------------------------------------------
SINGLE_NEEDLE_TEMPLATES = [
    {
        "needle": "The classified access code for Project Gemstone is RUBY-4821-QUARTZ.",
        "question": "What is the classified access code for Project Gemstone?",
        "answer": "RUBY-4821-QUARTZ",
    },
]

MULTI_NEEDLE_SET = [
    {
        "id": "needle_1",
        "needle": "The primary encryption key for the Meridian satellite network is FOXTROT-9173-LIMA.",
        "question": "What is the primary encryption key for the Meridian satellite network?",
        "answer": "FOXTROT-9173-LIMA",
    },
    {
        "id": "needle_2",
        "needle": "Dr. Elena Vasquez discovered that the high-pressure ignition threshold for compound ZR-8 is exactly 3,721 kilopascals.",
        "question": "What is the high-pressure ignition threshold for compound ZR-8?",
        "answer": "3,721 kilopascals",
    },
    {
        "id": "needle_3",
        "needle": "The coordinates for the backup data center in Reykjavik are 64.1466 degrees north latitude, 21.9426 degrees west longitude.",
        "question": "What are the coordinates for the backup data center in Reykjavik?",
        "answer": "64.1466 degrees north latitude, 21.9426 degrees west longitude",
    },
    {
        "id": "needle_4",
        "needle": "According to internal memo MX-2047, the quarterly budget allocation for Division Theta is 2.35 million euros.",
        "question": "What is the quarterly budget allocation for Division Theta according to memo MX-2047?",
        "answer": "2.35 million euros",
    },
    {
        "id": "needle_5",
        "needle": "The emergency shutdown sequence for reactor unit seven requires entering code BRAVO-TANGO-5590 followed by a biometric palm scan.",
        "question": "What is the emergency shutdown code for reactor unit seven?",
        "answer": "BRAVO-TANGO-5590",
    },
]


def build_haystack(target_chars: int, seed: int = 42) -> str:
    """Build filler text of approximately target_chars characters."""
    rng = random.Random(seed)
    paragraphs = list(HAYSTACK_PARAGRAPHS)
    text_parts = []
    current_len = 0
    idx = 0
    while current_len < target_chars:
        if idx % len(paragraphs) == 0 and idx > 0:
            rng.shuffle(paragraphs)
        para = paragraphs[idx % len(paragraphs)]
        text_parts.append(para)
        current_len += len(para) + 2  # +2 for \n\n
        idx += 1
    return "\n\n".join(text_parts)


def insert_needle_at_depth(haystack: str, needle: str, depth: float) -> str:
    """Insert needle at a given depth (0.0=start, 1.0=end) in the haystack.

    Inserts between paragraph boundaries to keep text natural.
    """
    paragraphs = haystack.split("\n\n")
    if len(paragraphs) <= 1:
        return needle + "\n\n" + haystack

    # Determine insertion point (paragraph index)
    insert_idx = int(depth * (len(paragraphs) - 1))
    insert_idx = max(0, min(insert_idx, len(paragraphs) - 1))

    paragraphs.insert(insert_idx, needle)
    return "\n\n".join(paragraphs)


def insert_needles_at_depths(
    haystack: str, needles: list[dict], n_needles: int
) -> tuple[str, list[dict]]:
    """Insert multiple needles spread evenly through the haystack.

    Returns the modified text and the list of inserted needle records.
    """
    paragraphs = haystack.split("\n\n")
    n = min(n_needles, len(needles))
    selected = needles[:n]

    # Spread needles evenly
    if n == 1:
        depths = [0.5]
    else:
        depths = [i / (n - 1) for i in range(n)]

    # Insert from back to front so indices stay valid
    insertions = []
    for needle_rec, depth in zip(selected, depths):
        idx = int(depth * (len(paragraphs) - 1))
        idx = max(0, min(idx, len(paragraphs) - 1))
        insertions.append((idx, needle_rec))

    # Sort by index descending for safe insertion
    insertions.sort(key=lambda x: x[0], reverse=True)
    placed = []
    for idx, needle_rec in insertions:
        paragraphs.insert(idx, needle_rec["needle"])
        placed.append(
            {
                "id": needle_rec["id"],
                "needle": needle_rec["needle"],
                "question": needle_rec["question"],
                "answer": needle_rec["answer"],
                "approximate_depth": round(
                    idx / max(len(paragraphs) - 1, 1), 3
                ),
            }
        )

    placed.reverse()  # restore original order
    return "\n\n".join(paragraphs), placed


def generate_single_needle_tests() -> list[dict]:
    """Generate single-needle test cases for all context lengths and depths."""
    needle_def = SINGLE_NEEDLE_TEMPLATES[0]
    tests = []

    for ctx_len in CONTEXT_LENGTHS:
        target_chars = ctx_len * CHARS_PER_TOKEN
        # Reserve some room for needle + question
        haystack_chars = target_chars - len(needle_def["needle"]) - 200
        haystack = build_haystack(haystack_chars)

        for depth in DEPTHS:
            context = insert_needle_at_depth(
                haystack, needle_def["needle"], depth
            )
            # Trim to target
            if len(context) > target_chars:
                # Trim from the end but keep needle intact
                context = context[:target_chars]

            tests.append(
                {
                    "context_length_tokens": ctx_len,
                    "context_length_label": f"{ctx_len // 1024}k",
                    "depth": depth,
                    "needle": needle_def["needle"],
                    "question": needle_def["question"],
                    "expected_answer": needle_def["answer"],
                    "context": context,
                    "prompt": (
                        f"Read the following text carefully and answer the question at the end.\n\n"
                        f"---BEGIN TEXT---\n{context}\n---END TEXT---\n\n"
                        f"Question: {needle_def['question']}\n"
                        f"Answer:"
                    ),
                }
            )
        print(f"  Single needle: {ctx_len // 1024}k done ({len(context):,} chars)")

    return tests


def generate_multi_needle_tests() -> list[dict]:
    """Generate multi-needle test cases for all context lengths."""
    tests = []

    # All context lengths get all 5 needles
    needle_counts = {
        2_048: 5,
        8_192: 5,
        24_576: 5,
        32_768: 5,
        65_536: 5,
        131_072: 5,
        262_144: 5,
    }

    for ctx_len in CONTEXT_LENGTHS:
        target_chars = ctx_len * CHARS_PER_TOKEN
        n_needles = needle_counts[ctx_len]
        total_needle_chars = sum(
            len(n["needle"]) for n in MULTI_NEEDLE_SET[:n_needles]
        )
        haystack_chars = target_chars - total_needle_chars - 400
        haystack = build_haystack(haystack_chars)

        context, placed_needles = insert_needles_at_depths(
            haystack, MULTI_NEEDLE_SET, n_needles
        )

        if len(context) > target_chars:
            context = context[:target_chars]

        questions = [n["question"] for n in placed_needles]
        answers = [n["answer"] for n in placed_needles]
        questions_text = "\n".join(
            f"{i + 1}. {q}" for i, q in enumerate(questions)
        )

        tests.append(
            {
                "context_length_tokens": ctx_len,
                "context_length_label": f"{ctx_len // 1024}k",
                "num_needles": n_needles,
                "needles": placed_needles,
                "questions": questions,
                "expected_answers": answers,
                "context": context,
                "prompt": (
                    f"Read the following text carefully and answer all questions at the end.\n\n"
                    f"---BEGIN TEXT---\n{context}\n---END TEXT---\n\n"
                    f"Questions:\n{questions_text}\n\n"
                    f"Answer each question on its own numbered line."
                ),
            }
        )
        print(
            f"  Multi needle: {ctx_len // 1024}k done "
            f"({n_needles} needles, {len(context):,} chars)"
        )

    return tests


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "niah")
    # Resolve to absolute
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("Generating single-needle NIAH tests...")
    single = generate_single_needle_tests()
    single_path = os.path.join(out_dir, "niah_single_needle.json")
    with open(single_path, "w") as f:
        json.dump(single, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(single)} test cases to {single_path}")
    size_mb = os.path.getsize(single_path) / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    print("\nGenerating multi-needle NIAH tests...")
    multi = generate_multi_needle_tests()
    multi_path = os.path.join(out_dir, "niah_multi_needle.json")
    with open(multi_path, "w") as f:
        json.dump(multi, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(multi)} test cases to {multi_path}")
    size_mb = os.path.getsize(multi_path) / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
