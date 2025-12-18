
from copy import deepcopy
from llm_attr_inf.dataset.base import Attribute, AttributeTypes


AGE = Attribute(
    name="age",
    description="age, in years, of the author at the time of writing the text snippet",
    short_description="the author's age",
    type=AttributeTypes.INT,
    reasonable_range=(10, 80)
)

AGE_ADULT = deepcopy(AGE)
AGE_ADULT.reasonable_range = (18, 80)

GENDER = Attribute(
    name="gender",
    description="sex of the author (male or female)",
    short_description="the author's gender",
    type=AttributeTypes.ENUM,
    enum_values=["male", "female"]
)

INCOME_CATEGORY = Attribute(
    name="income_category",
    description=("income level of the author. Choose from these options: less "
                 "than $30,000 USD, $30,000 − $60,000 USD, or more than $60,000 USD"),
    short_description="the author's income level",
    type=AttributeTypes.INT_ENUM,
    enum_values=[
        "less than $30k",
        "between $30k and $60k",
        "more than $60k"
    ]
)