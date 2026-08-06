# Cartographic CoT Annotation Protocol (v1.0)

## Objective

The objective of the reasoning trace is to teach the model **how to solve a cartographic reasoning problem**, rather than simply describing map contents.

Each reasoning step should correspond to **one explicit cartographic operation** that can be verified from the map.

---

# General Principles

## 1. Evidence-driven reasoning

Every reasoning step must be grounded in observable information from the map.

Allowed evidence includes:

- Legends
- Symbols
- Labels
- Colors
- Textures
- Boundaries
- Element shape
- Coastlines
- Terrain
- Coordinates
- Scale bars
- Compass directions

Do **not** use:

- World knowledge
- Assumptions
- Speculation
- Invisible information
- Subjective interpretation

---

## 2. Goal-driven reasoning

Reasoning should describe **how to solve the problem**, rather than simply describing the map.

Prefer action-oriented reasoning such as:

- Locate...
- Identify...
- Match...
- Compare...
- Trace...
- Measure...
- Transfer...
- Read...

Each sentence should help answer:

> **What operation should be performed next to solve the question?**

---

## 3. One cartographic operation per reasoning step

Avoid combining multiple reasoning operations into a single sentence.

Example:

Better:

> Locate the nearest town.
> Read its label.

Instead of: 
> Locate the nearest town and read its label.

---

## 4. Preserve intermediate reasoning

Do not jump directly from observations to the final answer.

Important intermediate reasoning should be preserved, including:

- Registration
- Localization
- Filtering
- Correspondence
- Comparison
- Measurement
- Label reading

---

## 5. Explicit map registration

Whenever multiple maps are involved, explicitly describe how the maps are aligned.

Registration evidence may include:

- Coastlines
- Rivers
- Road networks
- Administrative boundaries
- Country outlines
- Project boundaries
- Concession layouts
- Shared polygons
- Vegetation patterns
- Aerial imagery
- Coordinates
- Latitude / longitude
- Scale hierarchy

Avoid writing:

> The two maps correspond.

Instead explain **why** they correspond.

---

## 6. Preserve localization anchors

Keep intermediate spatial anchors whenever they help locate the target.

Examples include:

- Grid coordinates
- Map grid cells
- Concession numbers
- Quadrangles
- County names
- Nearby landmarks

---

## 7. Explicitly apply question constraints

Question constraints are part of the reasoning process.

Examples:

- excluding Fee Parcels
- excluding text in parentheses
- report the smallest value
- give all associated codes
- ignore touching but not crossing

---

## 8. Observation → Interpretation → Conclusion

Whenever legend interpretation is required, keep these three stages separated.

Example:

Observation

A neighboring polygon is colored yellow.

↓

Interpretation

According to the legend, the yellow polygon represents Barrick Gold Inc.

↓

Conclusion

The neighboring property is Barrick Gold Inc.

---

## 9. Prefer spatial operations over scene description

Reasoning should describe **what spatial operation is being performed**, rather than simply describing what is visible.

Better:

> Trace the shared boundary.

Instead of:

> A yellow polygon is beside a blue polygon.

---

# Task-specific Reasoning

---

## Within

Focus on:

- Locate the target region
- Identify the containing region
- Compare overlap
- Determine dominant region if required

Typical map elements:

- Legend
- Polygon
- Overlay
- Choropleth
- Administrative regions

---

## Distance

Focus on:

- Locate measurement objects
- Determine measurement type
- Identify measurement endpoints
- Estimate using the scale

Possible measurements include:

- Point ↔ Point
- Point ↔ Boundary
- Boundary ↔ Point
- Shared boundary length
- Straight-line distance
- Vertical distance
- Horizontal distance

Typical map elements:

- Scale bar
- Compass
- Coordinates
- Boundaries

---

## Orientation

Focus on:

- Locate the reference object
- Locate the target object
- Compare relative positions
- Determine direction using the compass

Typical map elements:

- Compass
- Relative positions
- Registration between maps

---

## Intersect

Focus on:

- Identify intersecting objects
- Apply intersection rules
- Distinguish crossing from touching
- Count valid intersections if required
- Identify categories on both sides if required

Typical map elements:

- Lines
- Polygons
- Roads
- Rivers
- Boundaries

---

## Equal (Correspondence)

Focus on:

- Identify the target entity or region
- Register multiple maps
- Transfer the entity or region
- Retrieve the corresponding attribute

Typical map elements:

- Coastlines
- Administrative regions
- Road networks
- Concession layouts
- Coordinates
- Aerial imagery

---

## Border

Focus on:

- Locate the target polygon
- Trace its shared boundary
- Identify adjacent polygons
- Retrieve neighboring attributes

Typical map elements:

- Polygon boundaries
- Shared borders
- Adjacent regions
- Labels
- Legends

---

# Recommended Reasoning Workflow

For most cartographic reasoning questions, the reasoning process follows:

1. Understand the question.

2. Interpret the legend (if required).

3. Locate the target object or region.

4. Register multiple maps (if applicable).

5. Perform the required spatial operation.

6. Apply question-specific constraints.

7. Read labels or attributes.

8. Produce the final answer.