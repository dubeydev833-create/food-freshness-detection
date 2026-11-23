"""Actionable Food Storage and Handling Recommendations for FreshCheck AI."""

from typing import Any, Dict, List, Optional


RECOMMENDATIONS_DATABASE: Dict[str, Dict[str, List[str]]] = {
    "Fruits": {
        "Fresh": [
            "Store in the crisper drawer of your refrigerator at 2°C to 4°C to slow ripening.",
            "Keep ethylene-producing fruits (e.g., apples, bananas) separate from ethylene-sensitive produce.",
            "Do not wash berries or grapes until immediately before consumption to avoid mold spore proliferation.",
            "Expected shelf life: 5 to 7 days under proper refrigeration.",
        ],
        "Moderate": [
            "Consume within 24 to 48 hours before quality degrades into decay.",
            "Consider processing into fruit smoothies, compotes, jams, or baking into pastries.",
            "Check peel surface for soft spots, bruising, or fungal colonization before consumption.",
            "Store refrigerated in an airtight container to isolate aromatic breakdown.",
        ],
        "Spoiled": [
            "DO NOT CONSUME. Discard immediately or compost in municipal green waste.",
            "Do not slice away fungal mold on soft fruits; hyphae roots penetrate deep into watery tissues.",
            "Wash and sanitize adjacent storage bins to prevent cross-contamination.",
            "Inspect nearby fruits for airborne spore transmission.",
        ],
    },
    "Vegetables": {
        "Fresh": [
            "Keep leafy greens in high-humidity crisper drawers wrapped in clean, dry paper towels.",
            "Store root vegetables (potatoes, onions) in a cool, dark, dry pantry away from sunlight.",
            "Optimal temperature for most fresh greens is 1°C to 3°C.",
            "Maintain ventilation; avoid sealing moist vegetables in unvented plastic bags.",
        ],
        "Moderate": [
            "Consume within 24 to 36 hours. Best suited for cooked dishes, soups, stir-fries, or stocks.",
            "If greens are slightly wilted but have no slime or sour odor, soak in cold water for 15 minutes.",
            "Blanch and freeze if immediate cooking is not planned.",
            "Inspect stems and leaves closely for microbial sliminess or foul breakdown.",
        ],
        "Spoiled": [
            "DO NOT CONSUME. High bacterial loads can cause foodborne gastrointestinal illness.",
            "Sliminess, black decay spots, and foul sulfurous odor indicate advanced putrefaction.",
            "Discard safely; do not feed spoiled vegetables to household pets.",
            "Sanitize the refrigerator crisper drawer thoroughly with soapy water and vinegar.",
        ],
    },
    "Meat": {
        "Fresh": [
            "Store on the lowest shelf of your refrigerator at <= 3°C (37°F) to prevent drip cross-contamination.",
            "Cook fresh poultry and ground meat within 1 to 2 days; roasts/steaks within 3 to 5 days.",
            "If not cooking immediately, wrap securely in freezer paper/vacuum seal and freeze at -18°C.",
            "Always wash cutting boards, knives, and hands with warm soapy water after handling.",
        ],
        "Moderate": [
            "Inspect texture and aroma carefully: any stickiness, sour smell, or discoloration requires disposal.",
            "Cook thoroughly to minimum internal safety temperatures (74°C / 165°F for poultry, 71°C for ground meat).",
            "Do not refreeze meat at this stage; cook immediately today.",
            "When in doubt, discard. Meat pathogens (Salmonella, Listeria) proliferate rapidly in warm ambient temperatures.",
        ],
        "Spoiled": [
            "CRITICAL HAZARD: DO NOT TASTE OR CONSUME UNDER ANY CIRCUMSTANCES.",
            "Spoiled meat produces heat-resistant bacterial toxins that cannot be eliminated by cooking.",
            "Double-bag and dispose of securely in outdoor trash to prevent pest access.",
            "Thoroughly bleach/disinfect all refrigerator shelves, prep counters, and utensils.",
        ],
    },
    "Dairy": {
        "Fresh": [
            "Keep stored at <= 4°C (40°F) in the coldest body of the refrigerator, not on door shelves.",
            "Keep cartons tightly sealed to prevent absorbing odors and oxidation.",
            "Consume milk within 5 to 7 days of opening; hard cheeses within 3 to 4 weeks.",
            "Ensure refrigeration is maintained immediately after each use.",
        ],
        "Moderate": [
            "Smell and taste a tiny drop: discard immediately if acidic, sour, or curdling is observed.",
            "Pasteurized milk nearing expiration can be used for baking, custards, or cooked puddings.",
            "Hard cheeses with minor surface mold can have mold cut off (1 inch margin), but discard soft cheeses with mold.",
            "Consume within 24 hours.",
        ],
        "Spoiled": [
            "DO NOT CONSUME. Sour odor, curdling, bubbling, or discoloration indicate spoilage.",
            "Spoiled dairy can harbor toxic molds and bacterial contaminants.",
            "Discard product immediately down drain with copious hot running water or into trash.",
            "Rinse and sanitize milk bottle caps, containers, and surrounding refrigerator area.",
        ],
    },
}


def get_storage_recommendations(
    category: str,
    predicted_class: str,
    temperature_celsius: Optional[float] = None,
    storage_condition: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve tailored storage recommendations and safety guidelines.

    Args:
        category: Food category ('Fruits', 'Vegetables', 'Meat', 'Dairy').
        predicted_class: Predicted class ('Fresh', 'Moderate', 'Spoiled').
        temperature_celsius: Optional observed ambient temperature.
        storage_condition: Optional current storage method.

    Returns:
        Dict[str, Any]: Structured tips, hazard warning, and temperature guidance.
    """
    cat_data = RECOMMENDATIONS_DATABASE.get(category, RECOMMENDATIONS_DATABASE["Fruits"])
    tips = cat_data.get(predicted_class, cat_data["Fresh"])

    # Temperature alert check
    temp_alert = None
    if temperature_celsius is not None:
        if predicted_class in ("Fresh", "Moderate") and temperature_celsius > 8.0 and category in ("Meat", "Dairy"):
            temp_alert = (
                f"⚠️ **Temperature Warning**: Observed temperature of {temperature_celsius}°C is within the food "
                f"safety Danger Zone (5°C - 60°C). {category} must be held below 4°C to prevent bacterial growth."
            )
        elif temperature_celsius > 25.0 and predicted_class == "Fresh":
            temp_alert = (
                f"🌡️ **High Ambient Temperature**: Storing at {temperature_celsius}°C accelerates enzymic "
                f"and microbial degradation. Transfer to a cool environment or refrigerator promptly."
            )

    return {
        "category": category,
        "predicted_class": predicted_class,
        "action_items": tips,
        "temperature_alert": temp_alert,
    }
