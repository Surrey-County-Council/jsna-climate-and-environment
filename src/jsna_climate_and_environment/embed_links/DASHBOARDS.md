
# Embeding external Dashboards
To embed an external dashboard in a clean and readable way you need to manually adjust the url that links to the published workbook. This is highly customizable but the 3 key features that help most are:

1. `:embed=y`

   this enables the dashboard to be embedded in another webpage
2. `:showVizHome=no`

   this removes the additional noise, making the dashboard look clean
3. `:size=100%,100%`

   this ensures the embedded dashboard is fitted exactly to the page

these settings are joined together using the ampersand sign. you end up with a static set of parameters that consistently show a clean dashboard link.

`?:embed=y&:showVizHome=no:size=100%,100%`

# Manually editing the url
Typically, url's from tableau public might look like the following examples:

- `https://public.tableau.com/views/IndicesofDeprivationIoD2025inSurrey/IndicesofDeprivation2025inSurrey?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link`

- `https://public.tableau.com/views/SurreyHousingdashboard2024/Affordabilitypriceandprivaterentals?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link`

Note the common root url: `https://public.tableau.com/views`

The dashboards will have a unique identifier and a name beffore the question mark. Anything after the question mark can be edited manually. Typically some defaults (language, lineage, credentials) are always set but we can ignore these after copying the url.

# Testing steps
1. Strip the question mark and everything after and manually past the url into a browser. ie. `https://public.tableau.com/views/SurreyHousingdashboard2024/Affordabilitypriceandprivaterentals`
2. note the unique identifier. ie. : `SurreyHousingdashboard2024/Affordabilitypriceandprivaterentals` 
3. replce everything after the question mark with our custom settings and paste into the browser. ie  `https://public.tableau.com/views/SurreyHousingdashboard2024/Affordabilitypriceandprivaterentals?:embed=y&:showVizHome=no:size=100%,100%`
4. if the link above works in your browser you can paste it directly into a Tableau weblink object in a .twbx file. Because of the common root and parameters, I have added a `Surrey-i-dashboards` parameter in tableau where you can copy and paste the unique identifier `step 1` and allow the user to choose the dashboard from a list of parameters.