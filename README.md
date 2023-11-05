# selling of units Pricing Dataset, melbourn

### `This is Supervised Task (Regression) `

### About Dataset

Context

This data was scraped from publicly available results posted every week from Domain.com.au, I've cleaned it as best I can, now it's up to make data analysis magic. The dataset includes Address, Type of Real estate, Suburb, Method of Selling, Rooms, Price, Real Estate Agent, Date of Sale and distance from C.B.D.

### About the project

- This is a machine learning project to predict unit/property selling price in Melbourn.
- This project aims to answers question about how much a unit price would be if given information such as location, number of bedrooms, etc? This would help potential tenant and also the owner to get the best price of their units, comparable to the market value.

### Content

There are 20 features with one unique ids (ads_id) and one target feature (monthly_rent)

-Suburb: Suburb

Address: Address

Rooms: Number of rooms

Price: Price in Australian dollars

Method:

- S - property sold;
- SP - property sold prior;
- PI - property passed in;
- PN - sold prior not disclosed;
- SN - sold not disclosed;
- NB - no bid;
- VB - vendor bid;
- W - withdrawn prior to auction;
- SA - sold after auction;
- SS - sold after auction price not disclosed.
- N/A - price or highest bid not available.

Type:

- br - bedroom(s);
- h - house,cottage,villa, semi,terrace;
- u - unit, duplex;
- t - townhouse;
- dev site - development site;
- o res - other residential.
- SellerG: Real Estate Agent

Date: Date sold

Distance: Distance from CBD in Kilometres

Regionname: General Region (West, North West, North, North east …etc)

Propertycount: Number of properties that exist in the suburb.

Bedroom2 : Scraped # of Bedrooms (from different source)

Bathroom: Number of Bathrooms

Car: Number of carspots

Landsize: Land Size in Metres

BuildingArea: Building Size in Metres

YearBuilt: Year the house was built

CouncilArea: Governing council for the area

Lattitude: Self explanitory

Longtitude: Self explanitory

- Inspiration

 in the past there was no easy way to understand whether certain unit pricing is making sense or not. With this dataset, I wanted to be able to answer the following questions:

- What are the biggest factor affecting the unit/rent pricing?
- here we want to analyze Mellbourne data to cover points:
- Which suburbs are the best to buy in?
- Which ones are value for money?
- Where's the expensive side of town?
- where should I buy a 2 bedroom unit?
