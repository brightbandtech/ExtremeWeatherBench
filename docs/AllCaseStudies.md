# Case studies for Extreme Weather Bench

We have collected data for high-impact weather across a variety of categories. The configuration files for each case study are available separately (see [Case study implementation details](CaseStudyYamlDetails.md)). Each category is described below and linked to the separate file that documents each case study in that category as well as the criteria used to choose case studies and any databases used to identify these cases. 

To collect the case studies, we looked at events worldwide in the time period of 2020-2024. For statistical significance, We aimed for at least 30 cases per category and sub-category but this is not always possible.

EWB is a community benchmark!  If you want to submit new categories or new case studies,  please make sure to add descriptions for any events you submit into these documents as well as  the information needed to download the data (see [Case study YAML details]( CaseStudyYamlDetails.md)).  

# EWB categories for v1 

| Main category     | Sub category          | Description                                         |
|-------------------|-----------------------|-----------------------------------------------------|
| [Severe Convective Weather](ConvectiveWx.md)    | [Overall severe days](SevereBustDays.md)   | Severe weather includes tornado outbreaks, major hail storms, and major wind storms  |
|                   | [High-risk bust days](SevereBustDays.md)   | Bust days for severe weather                        |
| [Heatwaves](Heatwaves.md)         | Land-based heat waves | Major heat waves worldwide that occurred over land  |
| [Tropical Cyclones](TropicalCyclones.md) |                       | Data and case studies related to hurricanes and tropical storms worldwide, broken out by basin       |
| [Flooding](Flooding.md)          | [Atmospheric Rivers](AtmosphericRivers.md)    | AR events worldwide. Other large-scale flooding will be in v2     |
| [Winter Weather](WinterWeather.md)    | [Major freezes](FreezeEvents.md)  | Large-scale freezing events that may or may not include other winter weather |

# Planned future updates

We have the events identified for the following full set of categories and future releases will include these as well as metrics for them. We welcome your input on this list! New categories are shown in bold below.


 Main category     | Sub category         | Description                                            |
|-------------------|---------------------|--------------------------------------------------------|
| Severe Convective Weather    | [Overall severe days](SevereBustDays.md)   | Severe weather includes tornado outbreaks, major hail storms, and major wind storms  |
|                   | [High-risk bust days](SevereBustDays.md)   | Bust days for severe weather                         |
|                   | [**Derechos**](Derechos.md)          | Derecho events                                       |
|                   | [**Major wind events**](WindEvents.md) | Major wind events excluding derechos but including cyclones  |
|                   | [**Tornado outbreaks**](TornadoOutbreaks.md) | Tornado outbreaks (with >= 20 tornadoes)             |
|                   | [**TC Tornado outbreaks**](TCTornadoOutbreaks.md) | Tornado outbreaks associated with a tropical cyclone            |
|                   | [**Hailstorms**](Hailstorms.md)        | Widespread hail events                               |
| [Heatwaves](Heatwaves.md)         | Land-based heat waves | Major heat waves worldwide that occurred over land   |
| [Tropical Cyclones](TropicalCyclones.md) |                       | Data and case studies related to hurricanes and tropical storms worldwide, broken out by basin       |
| [Flooding](Flooding.md)          | [Atmospheric Rivers](AtmosphericRivers.md)    | AR events worldwide. Other large-scale flooding will be in v2     |
|                   | [**TC related flooding**](TCFloods.md)| Major large-scale flooding from tropical cyclones   |
|                   | [**Other major floods**](OtherFloods.md) | Other non-TC and non-AR major flooding events       |
| Winter Weather    | [Major freezes](FreezeEvents.md)         | Large-scale freezing events that may or may not include other winter weather |
|                   | [**Major snowstorms**]((SnowEvents.md))  | Large-scale impactful snow events                    |
| **Wildfires**     | **Wildfire danger days**  | Days with high wildfire danger                   |
