"""
disease_data.py - Comprehensive Agricultural Disease Knowledge Base
Contains detailed disease descriptions, symptoms, severity ratings, causes,
organic treatments, chemical treatments, and prevention protocols for all 38
PlantVillage dataset classes supported by the MobileNetV2 CNN model.
"""

DISEASES_DB = {
    # ------------------ APPLE ------------------
    "Apple___Apple_scab": {
        "crop": "Apple",
        "disease": "Apple Scab",
        "scientific_name": "Venturia inaequalis",
        "type": "Fungal",
        "severity": "Moderate",
        "is_healthy": False,
        "symptoms": [
            "Olive-green to velvety dark brown lesions on the upper leaf surface",
            "Crinkled, distorted or curled foliage with premature leaf drop",
            "Dark, corky scabby spots developing on fruit skin reducing marketability"
        ],
        "cause": "Overwintering ascospores on dead leaves on the orchard floor, released in cool, wet spring rains (13-24°C).",
        "organic_treatment": [
            "Apply liquid copper or sulfur-based sprays before buds open",
            "Spray bio-fungicide containing Bacillus subtilis (Serenade)",
            "Shred and rake fallen leaves in autumn to disrupt spore fungal cycle"
        ],
        "chemical_treatment": [
            "Captan 50 WP or Mancozeb protectant sprays during early bud break",
            "Myclobutanil or Difenoconazole systemic sprays for curative action"
        ],
        "prevention": [
            "Plant resistant apple cultivars (e.g., Liberty, Enterprise, Freedom)",
            "Prune tree canopies annually to maximize sunlight and wind aeration",
            "Avoid overhead sprinkler irrigation that keeps leaves wet"
        ]
    },
    "Apple___Black_rot": {
        "crop": "Apple",
        "disease": "Black Rot (Frog-Eye Leaf Spot)",
        "scientific_name": "Botryosphaeria obtusa",
        "type": "Fungal",
        "severity": "Critical",
        "is_healthy": False,
        "symptoms": [
            "Frog-eye leaf spots: small purple specks expanding with tan centers and purple margins",
            "Mummified shriveled black fruits remaining attached to branches",
            "Reddish-brown sunken cankers on bark and woody limbs"
        ],
        "cause": "Fungus overwintering in dead twigs, mummified fruits, and bark cankers; spores splash-spread in warm rain.",
        "organic_treatment": [
            "Prune out dead wood, fire blight cankers, and mummified fruit during winter dormancy",
            "Apply copper sprays at silver tip to tight cluster stage"
        ],
        "chemical_treatment": [
            "Thiophanate-methyl or Captan sprays starting at petal fall",
            "Strobilurin fungicides (e.g., Kresoxim-methyl) in rotation"
        ],
        "prevention": [
            "Maintain strict orchard sanitation; burn or deep bury pruned debris",
            "Protect trees from winter injury and wood-boring insects"
        ]
    },
    "Apple___Cedar_apple_rust": {
        "crop": "Apple",
        "disease": "Cedar Apple Rust",
        "scientific_name": "Gymnosporangium juniperi-virginianae",
        "type": "Fungal (Heteroecious)",
        "severity": "Moderate",
        "is_healthy": False,
        "symptoms": [
            "Bright orange-yellow circular spots on upper leaf surfaces",
            "Small black fungal pycnia dots within the yellow spots",
            "Tube-like aecia structures protruding from the underside of leaves",
            "Early defoliation and fruit blemishes in severe infections"
        ],
        "cause": "Two-host rust fungus cycling between Eastern Red Cedar (Juniperus virginiana) and apple trees during warm, wet springs.",
        "organic_treatment": [
            "Apply sulfur or copper octanoate at pink bud stage through petal fall",
            "Remove eastern red cedar trees or galls within 1-2 km if feasible"
        ],
        "chemical_treatment": [
            "Myclobutanil (Immunox) applied from pink cluster to 3 weeks after petal fall",
            "Mancozeb or Propiconazole in rotation"
        ],
        "prevention": [
            "Select rust-resistant cultivars (e.g., Redfree, Priscilla, Enterprise)",
            "Inspect nearby juniper windbreaks and excise gelatinous brown galls"
        ]
    },
    "Apple___healthy": {
        "crop": "Apple",
        "disease": "Healthy Leaf",
        "scientific_name": "Malus domestica",
        "type": "None",
        "severity": "None",
        "is_healthy": True,
        "symptoms": [
            "Uniform vibrant green leaf blade without spots, blemishes, or lesions",
            "Smooth leaf margins and intact cuticle with healthy vein network",
            "Normal photosynthesis and active foliage growth"
        ],
        "cause": "Optimal orchard management, balanced nutrition, and effective preventive care.",
        "organic_treatment": ["Maintain current balanced organic orchard nutrition."],
        "chemical_treatment": ["None required."],
        "prevention": [
            "Annual dormant oil spray to control overwintering mites and scales",
            "Balanced N-P-K fertilization and regular canopy pruning"
        ]
    },

    # ------------------ BLUEBERRY ------------------
    "Blueberry___healthy": {
        "crop": "Blueberry",
        "disease": "Healthy Leaf",
        "scientific_name": "Vaccinium corymbosum",
        "type": "None",
        "severity": "None",
        "is_healthy": True,
        "symptoms": [
            "Deep green lustrous leaves with smooth borders",
            "No rust, powdery mildew, or necrotic marginal blight",
            "Vigorous new cane growth and healthy fruit clusters"
        ],
        "cause": "Acidic soil conditions (pH 4.5-5.2), adequate pine bark mulching, and proper drip irrigation.",
        "organic_treatment": ["Maintain acidic pine needle or pine bark mulch."],
        "chemical_treatment": ["None required."],
        "prevention": [
            "Monitor soil pH biannually and amend with elemental sulfur if necessary",
            "Maintain drip irrigation to avoid wet foliage"
        ]
    },

    # ------------------ CHERRY ------------------
    "Cherry_(including_sour)___Powdery_mildew": {
        "crop": "Cherry",
        "disease": "Powdery Mildew",
        "scientific_name": "Podosphaera clandestina",
        "type": "Fungal",
        "severity": "Moderate",
        "is_healthy": False,
        "symptoms": [
            "White talcum-like powdery fungal patches on young leaves and twigs",
            "Curled, puckered, or distorted upward leaf margins",
            "Stunted terminal shoot growth and unmarketable dull white-filmed cherries"
        ],
        "cause": "Airborne spores germinating in high relative humidity and warm temperatures (18-28°C), without requiring free surface water.",
        "organic_treatment": [
            "Potassium bicarbonate (MilStop) or horticultural mineral oil sprays",
            "Neem oil (Azadirachtin) foliar application at 7 to 10 day intervals"
        ],
        "chemical_treatment": [
            "Myclobutanil, Fenbuconazole, or Trifloxystrobin applied at petal fall",
            "Rotate chemical classes (FRAC 3 and FRAC 11) to avert resistance"
        ],
        "prevention": [
            "Prune dense interior water sprouts to optimize aeration and sunlight penetration",
            "Avoid excessive nitrogen fertilization that promotes overly succulent shoots"
        ]
    },
    "Cherry_(including_sour)___healthy": {
        "crop": "Cherry",
        "disease": "Healthy Leaf",
        "scientific_name": "Prunus cerasus",
        "type": "None",
        "severity": "None",
        "is_healthy": True,
        "symptoms": [
            "Glossy dark green leaves without chlorosis or shot-hole lesions",
            "Firm petioles and vigorous terminal spur development"
        ],
        "cause": "Optimal stone fruit orchard care, regular pruning, and disease-free rootstock.",
        "organic_treatment": ["Continue regular seasonal tree care."],
        "chemical_treatment": ["None required."],
        "prevention": [
            "Maintain post-harvest dormant copper sprays to protect bark lenticels"
        ]
    },

    # ------------------ CORN (MAIZE) ------------------
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "crop": "Corn (Maize)",
        "disease": "Cercospora & Gray Leaf Spot",
        "scientific_name": "Cercospora zeae-maydis",
        "type": "Fungal",
        "severity": "Critical",
        "is_healthy": False,
        "symptoms": [
            "Narrow, rectangular tan or gray lesions strictly bounded by leaf veins",
            "Lesions measuring 1-6 cm running parallel to leaf veins",
            "Severe blighting, premature foliage death, and stalk lodging"
        ],
        "cause": "Fungal survival in previous corn crop residue; favored by warm (25-32°C), humid overcast weather and dew.",
        "organic_treatment": [
            "Bio-fungicides based on Trichoderma or Bacillus species applied preventively",
            "Deep tillage of infected stubble to accelerate residue decomposition"
        ],
        "chemical_treatment": [
            "Azoxystrobin + Difenoconazole or Pyraclostrobin + Fluxapyroxad applied at VT-R1 growth stage"
        ],
        "prevention": [
            "Plant resistant corn hybrids (check regional resistance ratings)",
            "Practice 2-year crop rotation with non-host crops like soybean or clover"
        ]
    },
    "Corn_(maize)___Common_rust_": {
        "crop": "Corn (Maize)",
        "disease": "Common Rust",
        "scientific_name": "Puccinia sorghi",
        "type": "Fungal",
        "severity": "Moderate",
        "is_healthy": False,
        "symptoms": [
            "Golden-brown to cinnamon-brown powdery pustules on both upper and lower leaf surfaces",
            "Pustules rupture epidermal tissue releasing brick-red urediniospores",
            "Leaf chlorosis and chlorotic bands around pustule clusters"
        ],
        "cause": "Wind-blown spores transported northward from subtropical regions; favored by cool-moderate temps (16-25°C) and high humidity.",
        "organic_treatment": [
            "Sulfur dust or neem-based foliar sprays on sweet corn gardens",
            "Early harvesting of susceptible sweet corn before rust spreads"
        ],
        "chemical_treatment": [
            "Triazole fungicides (Propiconazole, Tebuconazole) or Strobilurins applied before 6 leaves show pustules"
        ],
        "prevention": [
            "Select corn varieties carrying Rp gene resistance",
            "Plant early in the season to evade late-season windborne spore flights"
        ]
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "crop": "Corn (Maize)",
        "disease": "Northern Leaf Blight",
        "scientific_name": "Exserohilum turcicum",
        "type": "Fungal",
        "severity": "Critical",
        "is_healthy": False,
        "symptoms": [
            "Long, elliptical, cigar-shaped grayish-green to tan lesions (2.5 to 15 cm)",
            "Dark olive fungal sporulation in concentric zones inside lesions during damp weather",
            "Extensive leaf burning causing ear fill failure"
        ],
        "cause": "Overwintering mycelia in crop residue, spreading via rain splash during moderate temps (18-27°C) and extended wetness.",
        "organic_treatment": [
            "Copper hydroxide sprays for small acreage organic corn",
            "Post-harvest residue management: discing or chopping stalks"
        ],
        "chemical_treatment": [
            "Strobilurin + Triazole premixes (Headline AMP, Quilt Xcel) applied at tasseling"
        ],
        "prevention": [
            "Plant corn hybrids with Ht gene resistance",
            "Rotate fields with non-grass crops for at least 1-2 seasons"
        ]
    },
    "Corn_(maize)___healthy": {
        "crop": "Corn (Maize)",
        "disease": "Healthy Leaf",
        "scientific_name": "Zea mays",
        "type": "None",
        "severity": "None",
        "is_healthy": True,
        "symptoms": [
            "Broad, elongated emerald green leaf blades without spots, rusts or blights",
            "Strong central midrib and healthy photosynthesizing canopy"
        ],
        "cause": "Optimal nitrogen supply, weed competition control, and disease-resistant hybrid seed.",
        "organic_treatment": ["Maintain balanced side-dress fertilization and soil aeration."],
        "chemical_treatment": ["None required."],
        "prevention": ["Rotate crops annually and maintain balanced soil fertility."]
    },

    # ------------------ GRAPE ------------------
    "Grape___Black_rot": {
        "crop": "Grape",
        "disease": "Black Rot",
        "scientific_name": "Guignardia bidwellii",
        "type": "Fungal",
        "severity": "Critical",
        "is_healthy": False,
        "symptoms": [
            "Small circular reddish-brown leaf spots with dark borders and black pimple-like pycnidia",
            "Infected berries turn pale, soften, turn pitch black, and shrivel into hard mummies",
            "Black elliptical cankers on shoots and vine tendrils"
        ],
        "cause": "Overwintering mummified berries on vines or soil; spores active in warm (21-32°C), wet conditions from bud break to veraison.",
        "organic_treatment": [
            "Copper sulfate + lime (Bordeaux mixture) applied early spring",
            "Thorough removal of all dried black berry mummies during winter pruning"
        ],
        "chemical_treatment": [
            "Mancozeb or Captan from 1-inch shoot growth through bloom",
            "Myclobutanil or Tebuconazole for systemic curative control"
        ],
        "prevention": [
            "Trellis training and summer canopy leaf pulling to maximize cluster airflow",
            "Destroy wild grapevines in nearby hedgerows"
        ]
    },
    "Grape___Esca_(Black_Measles)": {
        "crop": "Grape",
        "disease": "Esca (Black Measles)",
        "scientific_name": "Phaeomoniella chlamydospora & Fomitiporia mediterranea",
        "type": "Fungal Complex (Wood Decay)",
        "severity": "Critical",
        "is_healthy": False,
        "symptoms": [
            "Tiger-stripe interveinal chlorosis and necrosis across leaves",
            "Dark purple-brown spots ('measles') on white or red grape skin",
            "Internal dark wood streaking and white spongy wood rot in vine trunk cross-section"
        ],
        "cause": "Trunk-wound infecting fungal pathogens entering through pruning cuts and colonizing xylem vessels.",
        "organic_treatment": [
            "Pruning wound sealant with Trichoderma-based bio-protectants",
            "Surgical trunk renewal or vine retraining from lower sucker shoot"
        ],
        "chemical_treatment": [
            "Apply wound paste (Thiophanate-methyl) immediately over winter pruning cuts"
        ],
        "prevention": [
            "Prune vines late in the dormant season when wound healing is rapid",
            "Disinfect pruning shears regularly with 70% alcohol between cuts"
        ]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "crop": "Grape",
        "disease": "Leaf Blight (Isariopsis Leaf Spot)",
        "scientific_name": "Pseudocercospora vitis / Phaeoisariopsis vitis",
        "type": "Fungal",
        "severity": "Moderate",
        "is_healthy": False,
        "symptoms": [
            "Large, irregular dark reddish-brown to black spots on leaves",
            "Brown lesions causing leaves to dry, curl, and drop prematurely",
            "Post-harvest defoliation reducing vine carbohydrate reserves"
        ],
        "cause": "Pathogen overwintering on fallen vine leaves, spreading during late summer humid spells.",
        "organic_treatment": [
            "Apply copper oxychloride sprays after fruit harvest",
            "Rake and compost or burn fallen infected vineyard foliage"
        ],
        "chemical_treatment": [
            "Mancozeb, Carbendazim, or Azoxystrobin foliar sprays during post-bloom season"
        ],
        "prevention": [
            "Ensure proper vine canopy aeration through shoot positioning",
            "Maintain balanced potassium nutrition to improve foliar resistance"
        ]
    },
    "Grape___healthy": {
        "crop": "Grape",
        "disease": "Healthy Leaf",
        "scientific_name": "Vitis vinifera",
        "type": "None",
        "severity": "None",
        "is_healthy": True,
        "symptoms": [
            "Broad palmately lobed green leaves with supple texture and clear veins",
            "Clean grape clusters without shriveling, mildew, or berry cracking"
        ],
        "cause": "Proper vineyard trellis layout, controlled deficit irrigation, and disciplined canopy management.",
        "organic_treatment": ["Maintain straw or cover crop mulch along vine rows."],
        "chemical_treatment": ["None required."],
        "prevention": ["Annual winter cane pruning and balanced canopy aeration."]
    },

    # ------------------ ORANGE ------------------
    "Orange___Haunglongbing_(Citrus_greening)": {
        "crop": "Orange",
        "disease": "Citrus Greening (Huanglongbing)",
        "scientific_name": "Candidatus Liberibacter asiaticus",
        "type": "Bacterial (Phloem-limited)",
        "severity": "Critical",
        "is_healthy": False,
        "symptoms": [
            "Asymmetrical blotchy mottle leaf yellowing crossing leaf veins",
            "Thickened, corky veins with upright, small zinc-deficient leaves",
            "Small, lopsided fruit that stays green at the bottom and has a bitter taste",
            "Progressive canopy twig dieback and ultimate tree decline"
        ],
        "cause": "Phloem-restricted bacterium vectored by the Asian Citrus Psyllid (Diaphorina citri) or infected budwood.",
        "organic_treatment": [
            "Release parasitoid wasps (Tamarixia radiata) for biological psyllid suppression",
            "Nutritional foliar sprays (zinc, manganese, iron) to support tree vigor"
        ],
        "chemical_treatment": [
            "Imidacloprid or Thiamethoxam systemic soil drenches to suppress vector psyllids",
            "Rotational foliar insecticides (Beta-cyfluthrin, Spirotetramat)"
        ],
        "prevention": [
            "Plant only certified pathogen-free citrus nursery stock",
            "Scout weekly with yellow sticky cards for adult psyllid presence",
            "Eradicate infected trees in non-commercial settings to limit spread"
        ]
    },

    # ------------------ PEACH ------------------
    "Peach___Bacterial_spot": {
        "crop": "Peach",
        "disease": "Bacterial Spot",
        "scientific_name": "Xanthomonas arboricola pv. pruni",
        "type": "Bacterial",
        "severity": "Moderate",
        "is_healthy": False,
        "symptoms": [
            "Small, water-soaked polygonal angular spots near leaf tips",
            "Spot centers drop out giving a shot-hole appearance to leaves",
            "Foliar yellowing and premature defoliation",
            "Deep, pitted gummy cracks on peach fruit surface"
        ],
        "cause": "Overwintering bacteria in twig cankers and buds; favored by sandy soils, windy rainstorms, and warm temperatures (21-29°C).",
        "organic_treatment": [
            "Low-rate copper sprays (Copper octanoate) during dormancy and early leaf emergence",
            "Oxytetracycline formulations applied during bloom where licensed"
        ],
        "chemical_treatment": [
            "Copper bactericides at dormant bud swell",
            "Rotate with antibiotic sprays (Mycoshield) during cover spray periods"
        ],
        "prevention": [
            "Select resistant peach cultivars (e.g., Clayton, Candor, Harrow Beauty)",
            "Plant windbreaks to reduce sandblasting and leaf abrasions"
        ]
    },
    "Peach___healthy": {
        "crop": "Peach",
        "disease": "Healthy Leaf",
        "scientific_name": "Prunus persica",
        "type": "None",
        "severity": "None",
        "is_healthy": True,
        "symptoms": [
            "Lanceolate bright green leaves with serrated margins and no holes",
            "Clean bark and vigorous current-season shoot extension"
        ],
        "cause": "Well-drained soil, annual open-center pruning, and dormant wash applications.",
        "organic_treatment": ["Maintain regular organic stone fruit orchard management."],
        "chemical_treatment": ["None required."],
        "prevention": ["Apply dormant copper sprays in autumn after leaf fall."]
    },

    # ------------------ PEPPER (BELL) ------------------
    "Pepper,_bell___Bacterial_spot": {
        "crop": "Bell Pepper",
        "disease": "Bacterial Spot",
        "scientific_name": "Xanthomonas euvesicatoria",
        "type": "Bacterial",
        "severity": "Moderate",
        "is_healthy": False,
        "symptoms": [
            "Small, water-soaked circular to irregular dark brown lesions on leaves",
            "Lesions develop a pale brown center with prominent yellow halo",
            "Leaves turn chlorotic and shed rapidly, exposing fruit to sunscald",
            "Rough, raised warty scabs on pepper fruits"
        ],
        "cause": "Seed-borne or crop debris-borne bacteria spread by driving rain, overhead irrigation, and farm equipment handling wet plants.",
        "organic_treatment": [
            "Liquid copper octanoate / copper hydroxide mixed with bio-fungicide",
            "Hot-water seed soaking (50°C for 25 minutes) before planting"
        ],
        "chemical_treatment": [
            "Fixed copper tank-mixed with Mancozeb (synergistic bactericidal action)",
            "Plant activators like Acibenzolar-S-methyl (Actigard) to stimulate systemic acquired resistance"
        ],
        "prevention": [
            "Use certified disease-free seed and resistant hybrids (with Bs2 resistance)",
            "Strictly avoid working in pepper plots when leaves are wet from rain or dew",
            "3-year crop rotation away from solanaceous plants (tomato, potato, eggplant)"
        ]
    },
    "Pepper,_bell___healthy": {
        "crop": "Bell Pepper",
        "disease": "Healthy Leaf",
        "scientific_name": "Capsicum annuum",
        "type": "None",
        "severity": "None",
        "is_healthy": True,
        "symptoms": [
            "Broad, rich green leaves with glossy texture and unbroken margins",
            "Robust node development and healthy blossoming"
        ],
        "cause": "Optimum warm temperature (20-30°C), adequate calcium in soil, and drip irrigation.",
        "organic_treatment": ["Maintain consistent soil moisture and organic compost."],
        "chemical_treatment": ["None required."],
        "prevention": ["Mulch with black plastic or straw to preserve uniform root zone moisture."]
    },

    # ------------------ POTATO ------------------
    "Potato___Early_blight": {
        "crop": "Potato",
        "disease": "Early Blight",
        "scientific_name": "Alternaria solani",
        "type": "Fungal",
        "severity": "Moderate",
        "is_healthy": False,
        "symptoms": [
            "Brown to black circular spots with concentric target-board rings",
            "Yellowing around older bottom leaves spreading upward",
            "Dry, leathery, dark sunken lesions on potato tuber skin"
        ],
        "cause": "Spore-producing fungus thriving in alternating wet and dry periods, warm weather (24-29°C), and nutrient-stressed plants.",
        "organic_treatment": [
            "Copper hydroxide or copper oxychloride foliar sprays",
            "Bacillus subtilis biological sprays every 7-10 days"
        ],
        "chemical_treatment": [
            "Chlorothalonil or Mancozeb protectant sprays before canopy closure",
            "Boscalid, Difenoconazole, or Azoxystrobin in curative rotation"
        ],
        "prevention": [
            "Maintain balanced nitrogen and potassium fertilization to prevent premature vine senescence",
            "Plant certified seed tubers and practice 3-year crop rotation"
        ]
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "disease": "Late Blight",
        "scientific_name": "Phytophthora infestans",
        "type": "Oomycete / Water Mold",
        "severity": "Critical",
        "is_healthy": False,
        "symptoms": [
            "Dark water-soaked lesions on leaf tips and margins spreading rapidly",
            "White fungal fuzz on leaf undersides in humid, foggy conditions",
            "Stems develop dark brown to black rotting lesions",
            "Tubers exhibit a dry, reddish-brown granular rot beneath the skin"
        ],
        "cause": "Destructive water mold spreading via windborne sporangia in cool, wet weather (12-20°C and >90% humidity).",
        "organic_treatment": [
            "Fixed copper sprays applied early as protectant",
            "Destroy and bury all volunteer potato plants and cull piles"
        ],
        "chemical_treatment": [
            "Metalaxyl-M + Mancozeb (Ridomil Gold) systemic treatment",
            "Mandipropamid, Dimethomorph, or Cyazofamid applied preventatively"
        ],
        "prevention": [
            "Plant resistant potato varieties (e.g., Defender, Elba)",
            "Hill potatoes deeply to shield tubers from washed-down spores",
            "Kill vines 2-3 weeks before tuber harvest to prevent infection during lifting"
        ]
    },
    "Potato___healthy": {
        "crop": "Potato",
        "disease": "Healthy Leaf",
        "scientific_name": "Solanum tuberosum",
        "type": "None",
        "severity": "None",
        "is_healthy": True,
        "symptoms": [
            "Lush compound green leaves with crisp margins and sturdy petioles",
            "Vigorous vine canopy and strong tuber stolon formation"
        ],
        "cause": "Certified seed tubers, well-drained acidic soil (pH 5.0-6.0), and balanced hilling.",
        "organic_treatment": ["Maintain even watering and adequate hilling."],
        "chemical_treatment": ["None required."],
        "prevention": ["Scout fields twice weekly during cool, moist periods."]
    },

    # ------------------ RASPBERRY ------------------
    "Raspberry___healthy": {
        "crop": "Raspberry",
        "disease": "Healthy Leaf",
        "scientific_name": "Rubus idaeus",
        "type": "None",
        "severity": "None",
        "is_healthy": True,
        "symptoms": [
            "Vibrant green pinnate leaves with whitish undersides",
            "Firm canes without spur blights, cankers, or anthracnose lesions"
        ],
        "cause": "Good row spacing, cane thinning, and well-drained raised beds.",
        "organic_treatment": ["Maintain organic cane mulch and seasonal trellis training."],
        "chemical_treatment": ["None required."],
        "prevention": ["Prune spent floricanes immediately after fruiting to promote primocanes."]
    },

    # ------------------ SOYBEAN ------------------
    "Soybean___healthy": {
        "crop": "Soybean",
        "disease": "Healthy Leaf",
        "scientific_name": "Glycine max",
        "type": "None",
        "severity": "None",
        "is_healthy": True,
        "symptoms": [
            "Trifoliate leaves with deep green uniform color",
            "Active root nodulation and healthy pod cluster formation"
        ],
        "cause": "Effective Bradyrhizobium inoculation, crop rotation, and weed suppression.",
        "organic_treatment": ["Maintain good soil moisture and field drainage."],
        "chemical_treatment": ["None required."],
        "prevention": ["Rotate with non-legume crops to break soilborne fungal cycles."]
    },

    # ------------------ SQUASH ------------------
    "Squash___Powdery_mildew": {
        "crop": "Squash",
        "disease": "Powdery Mildew",
        "scientific_name": "Podosphaera xanthii",
        "type": "Fungal",
        "severity": "Moderate",
        "is_healthy": False,
        "symptoms": [
            "White, talcum powder-like fungal spots expanding over leaf surfaces and petioles",
            "Infected leaves turn yellow, brown, and dry up (senesce prematurely)",
            "Reduced fruit yield and sunburned, tasteless squash"
        ],
        "cause": "Airborne spores proliferating in high humidity and warm temperatures (20-27°C) with dense shade.",
        "organic_treatment": [
            "Spray potassium bicarbonate (3g/L) or diluted milk spray (1:9 with water)",
            "Apply neem oil or sulfur dust preventatively"
        ],
        "chemical_treatment": [
            "Myclobutanil, Quinoxyfen, or Cyflufenamid applied at first sign of white spots"
        ],
        "prevention": [
            "Select resistant squash and pumpkin cultivars",
            "Ensure wide spacing between hills to allow full sunlight and air circulation"
        ]
    },

    # ------------------ STRAWBERRY ------------------
    "Strawberry___Leaf_scorch": {
        "crop": "Strawberry",
        "disease": "Leaf Scorch",
        "scientific_name": "Diplocarpon earlianum",
        "type": "Fungal",
        "severity": "Moderate",
        "is_healthy": False,
        "symptoms": [
            "Irregular purplish-dark spots (1-5 mm) lacking white centers",
            "Spots coalesce until entire leaf turns brown, curled, and appears scorched or burned",
            "Calyx infection causing unmarketable dry brown caps on ripe strawberries"
        ],
        "cause": "Fungus overwintering in living leaves; conidia spread by rain splash in spring and summer.",
        "organic_treatment": [
            "Copper fungicide sprays applied at pre-bloom and post-harvest renovation",
            "Prune and burn heavily infected runner foliage"
        ],
        "chemical_treatment": [
            "Captan, Pyraclostrobin, or Cyprodinil + Fludioxonil applications"
        ],
        "prevention": [
            "Plant certified pathogen-free strawberry crowns in raised plastic mulch beds",
            "Mow and renovate June-bearing fields immediately following harvest"
        ]
    },
    "Strawberry___healthy": {
        "crop": "Strawberry",
        "disease": "Healthy Leaf",
        "scientific_name": "Fragaria × ananassa",
        "type": "None",
        "severity": "None",
        "is_healthy": True,
        "symptoms": [
            "Glossy trifoliate green leaves with clean serrated edges",
            "Vigorous runner production and plump, disease-free berries"
        ],
        "cause": "Drip irrigation, straw mulch, and healthy strawberry runner establishment.",
        "organic_treatment": ["Maintain clean straw or plastic mulch under fruit."],
        "chemical_treatment": ["None required."],
        "prevention": ["Replace strawberry beds every 3-4 years to avoid disease accumulation."]
    },

    # ------------------ TOMATO ------------------
    "Tomato___Bacterial_spot": {
        "crop": "Tomato",
        "disease": "Bacterial Spot",
        "scientific_name": "Xanthomonas perforans / euvesicatoria",
        "type": "Bacterial",
        "severity": "Moderate",
        "is_healthy": False,
        "symptoms": [
            "Small (less than 3mm), dark brown to black water-soaked angular spots on leaves",
            "Spots may coalesce causing leaf yellowing and ragged leaf margins",
            "Blister-like, scabby raised brown spots on green fruits with white halos"
        ],
        "cause": "Seed-borne or wind/rain-splashed bacteria thriving in warm (25-30°C) rainy seasons.",
        "organic_treatment": [
            "Liquid copper octanoate / copper hydroxide sprays mixed with Bacillus amyloliquefaciens",
            "Hot water seed treatment (50°C for 25 minutes) before sowing"
        ],
        "chemical_treatment": [
            "Copper bactericide combined with Mancozeb (enhances bacterial control efficacy)",
            "Streptomycin sulfate or Kasugamycin sprays where regionally licensed"
        ],
        "prevention": [
            "Purchase pathogen-free certified seeds and transplants",
            "Never cultivate, weed, or harvest tomato fields while foliage is wet",
            "Disinfect farm tools and stakes with 10% bleach solution"
        ]
    },
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "disease": "Early Blight",
        "scientific_name": "Alternaria solani",
        "type": "Fungal",
        "severity": "Moderate",
        "is_healthy": False,
        "symptoms": [
            "Brown to black concentric target-board like ring spots on older leaves",
            "Yellow halo (chlorosis) surrounding circular leaf lesions",
            "Premature defoliation starting from the lower canopy upwards",
            "Sunscald vulnerability on exposed fruits"
        ],
        "cause": "Fungal pathogen overwintering in plant debris, favored by warm temperatures (24-29°C) and prolonged leaf wetness or high humidity.",
        "organic_treatment": [
            "Apply copper-based organic fungicides or Bacillus subtilis sprays",
            "Spray neem oil formulation (5ml/L) at 7-day intervals",
            "Prune and dispose of infected bottom leaves immediately (do not compost)"
        ],
        "chemical_treatment": [
            "Chlorothalonil (0.2%) or Mancozeb (0.25%) applied at first sign of symptoms",
            "Azoxystrobin or Difenoconazole in rotation to prevent fungicide resistance"
        ],
        "prevention": [
            "Practice minimum 3-year crop rotation with non-solanaceous crops",
            "Use drip irrigation rather than overhead sprinklers to keep foliage dry",
            "Mulch soil around plants to prevent spores splashing up from soil onto foliage",
            "Ensure proper plant spacing for optimal air circulation"
        ]
    },
    "Tomato___Late_blight": {
        "crop": "Tomato",
        "disease": "Late Blight",
        "scientific_name": "Phytophthora infestans",
        "type": "Oomycete / Water Mold",
        "severity": "Critical",
        "is_healthy": False,
        "symptoms": [
            "Large, dark water-soaked greasy lesions on leaves and stems",
            "White velvety fungal/mildew growth on leaf undersides in humid conditions",
            "Rapid collapse and browning of foliage resembling frost burn",
            "Firm, rough dark brown rot on green or ripe fruits"
        ],
        "cause": "Destructive water-mold pathogen spreading rapidly through wind-borne sporangia during cool (15-20°C), foggy, or wet rainy weather.",
        "organic_treatment": [
            "Fixed copper sprays applied early before widespread outbreak",
            "Bio-fungicides containing Trichoderma harzianum soil inoculation",
            "Immediate rogueing: remove and bury entire severely infected plants"
        ],
        "chemical_treatment": [
            "Systemic fungicides: Metalaxyl-M + Mancozeb (Ridomil Gold) at 2.5g/L",
            "Dimethomorph or Cymoxanil alternating with protectant fungicides"
        ],
        "prevention": [
            "Plant certified disease-free transplants and resistant cultivars (e.g., Mountain Magic, Defiant)",
            "Eliminate cull piles and volunteer tomato and potato plants nearby",
            "Strictly avoid overhead irrigation during humid spells",
            "Provide ample ventilation in greenhouse tunnels"
        ]
    },
    "Tomato___Leaf_Mold": {
        "crop": "Tomato",
        "disease": "Leaf Mold",
        "scientific_name": "Passalora fulva (Cladosporium fulvum)",
        "type": "Fungal",
        "severity": "Moderate",
        "is_healthy": False,
        "symptoms": [
            "Pale green to yellow spots with indistinct margins on upper leaf surfaces",
            "Olive-green to velvety brown mold patches on the corresponding lower surface",
            "Leaves curl, wilt, turn brown, and drop prematurely in severe cases"
        ],
        "cause": "High relative humidity (>85%) and warm temperatures (22-26°C), extremely common in protected greenhouse polyhouses.",
        "organic_treatment": [
            "Spray copper hydroxide or potassium bicarbonate on both leaf surfaces",
            "Increase greenhouse ventilation to reduce ambient relative humidity below 80%"
        ],
        "chemical_treatment": [
            "Chlorothalonil, Mancozeb, or Difenoconazole applied at early detection"
        ],
        "prevention": [
            "Select leaf mold-resistant tomato hybrids (Cf gene resistance)",
            "Space plants widely and prune lower suckers to maximize cross-breeze"
        ]
    },
    "Tomato___Septoria_leaf_spot": {
        "crop": "Tomato",
        "disease": "Septoria Leaf Spot",
        "scientific_name": "Septoria lycopersici",
        "type": "Fungal",
        "severity": "Moderate",
        "is_healthy": False,
        "symptoms": [
            "Numerous small circular spots (1.5-3mm) with dark brown borders and gray/tan centers",
            "Tiny black specks (pycnidia) visible inside mature spot centers",
            "Leaves turn yellow, curl, and wither, progressing from bottom to top"
        ],
        "cause": "Overwintering in solanaceous weeds and previous tomato debris; spores splash onto lower leaves in wet conditions (20-25°C).",
        "organic_treatment": [
            "Copper fungicide sprays applied at 7-10 day intervals",
            "Strip lower leaves up to 18 inches off the ground to prevent soil splash"
        ],
        "chemical_treatment": [
            "Chlorothalonil, Mancozeb, or Azoxystrobin applications"
        ],
        "prevention": [
            "Apply 2-3 inches of clean organic straw mulch around plant base",
            "Eradicate nightshade and horsenettle weeds around field perimeters"
        ]
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "crop": "Tomato",
        "disease": "Two-Spotted Spider Mite Infestation",
        "scientific_name": "Tetranychus urticae",
        "type": "Pest (Arachnid Acari)",
        "severity": "Moderate",
        "is_healthy": False,
        "symptoms": [
            "Fine yellow or white stippling and speckled speckles across upper leaf surfaces",
            "Delicate silken webbing beneath leaves and around stem junctions",
            "Foliage bronze, dry, brittle, and drop off in heavy infestations"
        ],
        "cause": "Microscopic sap-feeding arachnids proliferating exponentially in hot, dry, dusty weather (>30°C).",
        "organic_treatment": [
            "Release predatory mites (Phytoseiulus persimilis or Neoseiulus californicus)",
            "Spray insecticidal potassium salts or pure cold-pressed neem oil (5ml/L)",
            "Wash foliage undersides with pressurized water spray to disrupt webs"
        ],
        "chemical_treatment": [
            "Abamectin, Spiromesifen, or Bifenazate selective miticide sprays",
            "Avoid broad-spectrum pyrethroids which kill beneficial predatory mites"
        ],
        "prevention": [
            "Keep farm tracks and greenhouse alleyways watered to suppress dust",
            "Maintain overhead misting to raise humidity in dry polyhouses"
        ]
    },
    "Tomato___Target_Spot": {
        "crop": "Tomato",
        "disease": "Target Spot",
        "scientific_name": "Corynespora cassiicola",
        "type": "Fungal",
        "severity": "Moderate",
        "is_healthy": False,
        "symptoms": [
            "Small brown circular spots with concentric rings resembling a target",
            "Spots lack prominent yellow halo distinguishing them from early blight",
            "Sunken dark circular lesions on green and ripening tomato fruits"
        ],
        "cause": "Fungal pathogen favored by warm temperatures (20-28°C) and extended dew/wetness periods in humid subtropical zones.",
        "organic_treatment": [
            "Copper hydroxide sprays mixed with Bacillus amyloliquefaciens",
            "Remove and discard severely spotted lower foliage"
        ],
        "chemical_treatment": [
            "Fluxapyroxad + Pyraclostrobin, Boscalid, or Chlorothalonil sprays"
        ],
        "prevention": [
            "Staking and tying tomatoes to keep vines off wet soil",
            "Maintain wide plant spacing and drip irrigation"
        ]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "crop": "Tomato",
        "disease": "Yellow Leaf Curl Virus",
        "scientific_name": "Tomato Yellow Leaf Curl Begomovirus (TYLCV)",
        "type": "Viral",
        "severity": "Critical",
        "is_healthy": False,
        "symptoms": [
            "Severe upward curling and cupping of leaflets",
            "Marked interveinal yellowing (chlorosis) and purple veins beneath leaf",
            "Stunted bushy dwarf plant stature with short internodes",
            "Dramatic floral drop and near total failure of fruit set"
        ],
        "cause": "Begomovirus transmitted persistently by the Silverleaf Whitefly (Bemisia tabaci); not seed-borne or mechanically transmissible.",
        "organic_treatment": [
            "Install 50-mesh insect-proof netting in nursery seedling screens",
            "Deploy yellow sticky cards (1 card per 100 sq meters) to trap whitefly vectors",
            "Spray neem oil formulation or Beauveria bassiana entomopathogenic fungus",
            "Rogue out and destroy infected viral plants immediately"
        ],
        "chemical_treatment": [
            "Systemic neonicotinoids (Imidacloprid, Thiamethoxam) at transplanting",
            "Spiromesifen or Cyantraniliprole foliar spray for whitefly nymph management"
        ],
        "prevention": [
            "Grow certified TYLCV-resistant hybrids (e.g., Tycoon, Securitas)",
            "Maintain 2-month host-free bridge period between successive solanaceous crops"
        ]
    },
    "Tomato___Tomato_mosaic_virus": {
        "crop": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "scientific_name": "Tomato Mosaic Tobamovirus (ToMV)",
        "type": "Viral",
        "severity": "Critical",
        "is_healthy": False,
        "symptoms": [
            "Mottling of leaves with alternating light and dark green mosaic patterns",
            "Shoestring leaf distortion and blistered, puckered foliage",
            "Stunted plant growth and internal brown necrosis in green fruit walls"
        ],
        "cause": "Extremely stable tobamovirus mechanically transmitted through handling, pruning shears, contaminated seed, and tobacco residues.",
        "organic_treatment": [
            "Dip hands and pruning tools in non-fat dry milk solution (20%) or 10% TSP",
            "Immediately rogue and incinerate symptomatic virus-infected plants"
        ],
        "chemical_treatment": [
            "No chemical viricides exist; manage hygiene and prevent vector transmission"
        ],
        "prevention": [
            "Plant ToMV-resistant cultivars (marked with Tm-2 or Tm-2^2 gene)",
            "Prohibit tobacco smoking or chewing near tomato greenhouses",
            "Steam sterilize potting soil and greenhouse propagation trays"
        ]
    },
    "Tomato___healthy": {
        "crop": "Tomato",
        "disease": "Healthy Leaf",
        "scientific_name": "Solanum lycopersicum",
        "type": "None",
        "severity": "None",
        "is_healthy": True,
        "symptoms": [
            "Rich deep green compound leaves with sharp defined margins",
            "Sturdy stems, active apical growth, and abundant floral clusters",
            "Firm, glossy, unblemished tomato fruits ripening uniformly"
        ],
        "cause": "Optimal horticultural balance: adequate sunlight, drip irrigation, balanced N-P-K, and preventive IPM scouting.",
        "organic_treatment": ["Maintain compost tea drench and balanced mycorrhizal root health."],
        "chemical_treatment": ["None required."],
        "prevention": [
            "Support with sturdy stakes or trellises to keep foliage away from soil",
            "Mulch with straw or black plastic film and water strictly at root base"
        ]
    }
}

# Aliases to guarantee backward compatibility with previous class naming variations
ALIASES = {
    "Corn___Common_rust": "Corn_(maize)___Common_rust_",
    "Corn___Northern_Leaf_Blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn___healthy": "Corn_(maize)___healthy",
    "Pepper_bell___Bacterial_spot": "Pepper,_bell___Bacterial_spot",
    "Pepper_bell___healthy": "Pepper,_bell___healthy",
    "Tomato___Yellow_Leaf_Curl_Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Cherry___Powdery_mildew": "Cherry_(including_sour)___Powdery_mildew",
    "Cherry___healthy": "Cherry_(including_sour)___healthy",
}

for alias, canonical in ALIASES.items():
    if canonical in DISEASES_DB and alias not in DISEASES_DB:
        DISEASES_DB[alias] = DISEASES_DB[canonical]


def get_disease_info(class_name: str) -> dict:
    """
    Retrieve comprehensive agronomic disease information by model class name.
    Includes automatic alias resolution and intelligent fallback.
    """
    if class_name in DISEASES_DB:
        return DISEASES_DB[class_name]

    # Resolve alias if present
    if class_name in ALIASES and ALIASES[class_name] in DISEASES_DB:
        return DISEASES_DB[ALIASES[class_name]]

    # Normalized lookup (ignore punctuation / underscores)
    normalized = class_name.lower().replace("_", "").replace(",", "").replace("(", "").replace(")", "").replace(" ", "")
    for k, v in DISEASES_DB.items():
        k_norm = k.lower().replace("_", "").replace(",", "").replace("(", "").replace(")", "").replace(" ", "")
        if normalized == k_norm:
            return v

    # Fallback parsing
    crop = class_name.split("___")[0].replace("_", " ").replace("(", "").replace(")", "").strip() if "___" in class_name else "Plant"
    disease = class_name.split("___")[-1].replace("_", " ").strip() if "___" in class_name else class_name
    is_healthy = "healthy" in class_name.lower()
    return {
        "crop": crop,
        "disease": disease,
        "scientific_name": f"{crop} spec.",
        "type": "Normal" if is_healthy else "Pathogenic Condition",
        "severity": "None" if is_healthy else "Moderate",
        "is_healthy": is_healthy,
        "symptoms": ["Normal green leaf structure with uniform pigmentation."] if is_healthy else ["Visible leaf lesions, discoloration or foliar distress."],
        "cause": "Optimal plant health and balanced nutrition." if is_healthy else "Suspected fungal, bacterial, or environmental foliar pathogen.",
        "organic_treatment": ["Maintain routine organic care and balanced soil irrigation."] if is_healthy else ["Apply organic copper or neem oil spray.", "Remove and dispose of severely damaged leaves."],
        "chemical_treatment": ["None needed."] if is_healthy else ["Apply broad-spectrum protectant fungicide if condition progresses."],
        "prevention": ["Maintain balanced watering, adequate sun exposure, and weekly scouting."]
    }


def get_all_classes() -> list:
    """Return list of all canonical 38 supported classes."""
    return list(DISEASES_DB.keys())
