# Case studies for Extreme Weather Bench

We have collected data for high-impact weather across a variety of categories. The configuration files for each case study are available separately (see [Case study implementation details](CaseStudyYamlDetails.md)). Each category is described below and linked to the separate file that documents each case study in that category as well as the criteria used to choose case studies and any databases used to identify these cases. 

To collect the case studies, we looked at events worldwide in the time period of 2020-2024. For statistical significance, We aimed for at least 30 cases per category and sub-category but this is not always possible.

EWB is a community benchmark!  If you want to submit new categories or new case studies,  please make sure to add descriptions for any events you submit into these documents as well as  the information needed to download the data (see [Case study YAML details]( CaseStudyYamlDetails.md)).  

# EWB categories 

## Severe Convective Weather

Severe convective weather includes tornado outbreaks, major hail storms, and major wind storms. Most of these are not able to be resolved at the current resolution of global models. We have collected case studies for multiple high-impact convective weather phenomena and provide each of these below. For the initial release, we provide data only for overall severe convective days and high-risk bust days.

The overall description of what data sources we used as well as how we chose the different cases is [here](ConvectiveWx.md).

### Case studies and data available in v1
* [Overall severe days](OverallSevereDays.md) 
* [High-risk bust days](SevereBustDays.md) 

### Cases studies available now, data in future release
* [Derechos](Derechos.md) 
* [Major wind events](WindEvents.md) excluding derechos but including cyclones
* [Tornado outbreaks](TornadoOutbreaks.md) includes major outbreak days by excludes any caused by tropical cyclones
* [TC Tornado outbreaks](TCTornadoOutbreaks.md) includes major outbreaks associated with a tropical cyclone
* [Large-scale Hail outbreaks](Hailstorms.md)

## Heatwaves 

Heatwaves are growing in frequency and intensity. In v1, we provide case studies and data on land-based heatwaves. If there is interest in the future, we could expand this to include ocean-based heatwaves.

### Case studies and data available in v1
* [Land-based heatwaves](Heatwaves.md) 

## Tropical cyclones

We have provided case studies and data for tropical cyclones, broken out by basin. 

### Case studies and data available in v1

* [Tropical Cyclones](TropicalCyclones.md)

## Flooding

Major flooding has many causes.  For v1, we provide case studies and data for atmospheric rivers. We have collected case studies for tropical cyclone related flooding as well as other major floods and the data for these will be available in a future release.

The overall description of what data sources we used as well as how we chose the different cases is [here](Flooding.md).

### Case studies and data available in v1

* [Atmospheric Rivers](AtmosphericRivers.md)

### Cases studies available now, data in future release

* [TC related flooding](TCFloods.md) - Major large-scale flooding from tropical cyclones 
* [Other major floods](OtherFloods.md) - Other non-TC and non-AR major flooding events

## Winter weather

Winter weather causes a variety of major impacts to society.  We have collected case studies for two major phenomena, major freeze events and major snowstorms.  

### Case studies and data available in v1

* [Major freezes](FreezeEvents.md)- Large-scale freezing events that may or may not include other winter weather

### Cases studies available now, data in future release

* [Major snowstorms]((SnowEvents.md)) - Large-scale impactful snow events   

