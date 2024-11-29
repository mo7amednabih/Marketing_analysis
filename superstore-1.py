import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display option to show all columns
pd.set_option("display.max_columns", None)

# Load the dataset
df = pd.read_csv("Superstore Sales Dataset.csv", sep=",", encoding="latin1")

# Preprocessing

# Drop rows with missing values
df = df.dropna()

# Drop unnecessary columns: 'Row ID' and 'Order ID'
df = df.drop(["Row ID", "Order ID"], axis=1)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Convert 'Order Date' and 'Ship Date' to datetime format (day comes first)
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)

# Extract year, month, and day from 'Order Date'
df['Order Year'] = df['Order Date'].dt.year
df['Order Month'] = df['Order Date'].dt.month
df['Order Day'] = df['Order Date'].dt.day

# Extract year, month, and day from 'Ship Date'
df['Ship Year'] = df['Ship Date'].dt.year
df['Ship Month'] = df['Ship Date'].dt.month
df['Ship Day'] = df['Ship Date'].dt.day

# Calculate shipping time (difference between 'Ship Date' and 'Order Date' in days)
df['Shipping Time'] = (df['Ship Date'] - df['Order Date']).dt.days

# Optional: Check the processed data (uncomment to use)
# print(df.info())
# print(df.describe())
# print(df.duplicated().sum())
# print(df.head())



# Dashboard 1: Insights Explore

# 1. Top 10 Most Sold Products
# This gives us insight into the most popular products based on the number of sales.
top_10_products = df["Product Name"].value_counts().head(10)

# 2. Top 10 Products Generating the Most Sales
# These are the products that generated the highest revenue.
top_10_refunded_products = df.groupby("Product Name")[['Sales']].sum().sort_values('Sales', ascending=False).head(10)
top_10_refunded_sales = top_10_refunded_products['Sales']

# 3. Top Selling Product Categories
# This shows us which categories contribute the most to sales.
top_selling_categories = df['Category'].value_counts()

# Visualizing the Results
# Plot: Top 10 Most Sold Products
# A bar chart showing the top 10 most sold products.
plt.figure(figsize=(10, 6))
plt.bar(top_10_products.index, top_10_products.values, color='skyblue')
plt.title('Top 10 Most Sold Products')
plt.xlabel('Product Name')
plt.ylabel('Sales Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot: Top 10 Products Generating the Most Sales
# A colorful bar chart representing the top 10 products based on revenue.
colors = plt.cm.Paired(np.arange(len(top_10_refunded_sales)))
plt.figure(figsize=(10, 6))
bars = plt.bar(top_10_refunded_sales.index, top_10_refunded_sales.values, color=colors)

# Adding a title and axis labels
plt.title('Top 10 Products Generating the Most Sales')
plt.xlabel('Product Name')
plt.ylabel('Sales Amount')

# Hiding the labels under the bars
plt.xticks(ticks=np.arange(len(top_10_refunded_sales)), labels=['']*len(top_10_refunded_sales))

# Adding a legend to identify each product by its color
plt.legend(bars, top_10_refunded_sales.index, title="Product Name", bbox_to_anchor=(1.05, 1), loc='upper left')

# Improving layout
plt.tight_layout()

# Displaying the plot
plt.show()

# Plot: Top Selling Product Categories
# A pie chart that shows the contribution of different categories to the overall sales.
plt.figure(figsize=(8, 8))
plt.pie(top_selling_categories, labels=top_selling_categories.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Top Selling Product Categories')
plt.tight_layout()
plt.show()










# Dashboard 2: Shipping, Regional, and Sales Time Analysis
# Setting the figure size for the entire dashboard
plt.figure(figsize=(10, 7))

### 1. Most Used Shipping Method
plt.subplot(2, 2, 1)  # First plot (top-left)
df['Ship Mode'].value_counts().plot(kind="pie", autopct="%1.0f%%", 
                                    colors=['skyblue', 'lightgreen', 'lightcoral', 'orange'])
plt.title('Most Used Shipping Method')

### 2. Monthly Sales Trends Over the Years
# Group by Year and Month to calculate monthly sales
monthly_sales = df.groupby(['Order Year', 'Order Month'])['Sales'].sum().reset_index()

# Plotting Sales Trends
plt.subplot(2, 2, 2)  # Second plot (top-right)
sns.lineplot(data=monthly_sales, x='Order Month', y='Sales', hue='Order Year', marker='o', palette='tab10')
plt.title('Monthly Sales Trends Over the Years')
plt.xlabel('Order Month')
plt.ylabel('Total Sales')

### 3. Average Shipping Time by Year
# Calculate average shipping time by year
average_shipping_time_by_year = df.groupby('Order Year')['Shipping Time'].mean().reset_index()

# Plotting Average Shipping Time
plt.subplot(2, 1, 2)  # Third plot (bottom, full row)
bars = plt.bar(average_shipping_time_by_year['Order Year'], average_shipping_time_by_year['Shipping Time'], 
               color='skyblue', width=0.4)

# Adding values on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 1), ha='center', va='bottom')

# Adding title and labels
plt.title('Average Shipping Time by Year')
plt.xlabel('Year')
plt.ylabel('Average Shipping Time (days)')
plt.xticks(average_shipping_time_by_year['Order Year'], rotation=0)
plt.grid(axis='y')  # Adding gridlines for better readability

# Final layout adjustments
plt.tight_layout()
plt.show()

# # ------------------------------
# # Regional Sales Dashboard
# # ------------------------------

# Setting the figure size for the second dashboard
plt.figure(figsize=(10, 7))

### 1. Sales by Region
region_sales = df.groupby('Region')['Sales'].sum().reset_index()
plt.subplot(2, 2, 1)  # First plot (top-left)
sns.barplot(x='Sales', y='Region', data=region_sales, palette="coolwarm")
plt.title('Sales by Region')
plt.xlabel('Total Sales')
plt.ylabel('Region')

### 2. Top 10 Cities by Sales
top_cities = df.groupby("City")[['Sales']].sum().sort_values('Sales', ascending=False).head(10)
plt.subplot(2, 2, 2)  # Second plot (top-right)
top_cities.plot(kind="bar", legend=False, color='lightblue', ax=plt.gca())
plt.title('Top 10 Cities by Sales')
plt.xlabel('City')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)  # Rotate city names for better readability

### 3. Top 10 States by Sales
top_states = df.groupby("State")[['Sales']].sum().sort_values('Sales', ascending=False).head(10)
plt.subplot(2, 1, 2)  # Third plot (bottom, full row)
top_states.plot(kind="bar", legend=False, color='lightgreen', ax=plt.gca())
plt.title('Top 10 States by Sales')
plt.xlabel('State')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)  # Rotate state names for better readability

# Final layout adjustments
plt.tight_layout()
plt.show()




#Dashboard 3: Customer Insights
# Setting the figure size for the entire dashboard
plt.figure(figsize=(10, 5))

### 1. Sales by Customer Segment
# Grouping data by customer segment to calculate total sales
segment_sales = df.groupby('Segment')['Sales'].sum().reset_index()

plt.subplot(2, 2, 1)  # First plot (top-left)
sns.barplot(x='Sales', y='Segment', data=segment_sales, palette="Set2")
plt.title('Sales by Customer Segment')
plt.xlabel('Total Sales')
plt.ylabel('Segment')

### 2. Top 10 Customers by Average Sales
# Grouping data by customer ID and name to calculate average sales
avg_sales = df.groupby(["Customer ID", "Customer Name"])[['Sales']].agg(["mean"]).reset_index()

# Rename the column to remove MultiIndex
avg_sales.columns = ['Customer ID', 'Customer Name', 'Mean Sales']

# Sorting the results by average sales and selecting the top 10 customers
top_customers = avg_sales.sort_values('Mean Sales', ascending=False).head(10)

# Plotting the top 10 customers by average sales
plt.subplot(2, 2, 2)  # Second plot (top-right)
top_customers.plot(x='Customer Name', y='Mean Sales', kind="bar", legend=False, color='lightblue', ax=plt.gca())
plt.title('Top 10 Customers by Average Sales')
plt.ylabel('Average Sales')
plt.xticks(rotation=45)  # Rotate customer names for better readability

# Final layout adjustments
plt.tight_layout()
plt.show()





#Dashboard 4: Shipping Time Analysis Documentation
# Identify delayed orders (e.g., those with a shipping time greater than 5 days)
delayed_orders = df[df['Shipping Time'] > 5]

# Analyze potential causes of delays based on various factors
# Compare average shipping time by ship mode
shipping_mode_delay = delayed_orders.groupby('Ship Mode')['Shipping Time'].mean()
# Compare average shipping time by region
region_delay = delayed_orders.groupby('Region')['Shipping Time'].mean()

# Compare average shipping time by product category
category_delay = delayed_orders.groupby('Category')['Shipping Time'].mean()

# Set up the size of the dashboard
plt.figure(figsize=(10, 8))  # Size suitable for three plots

# --- 1. Average Shipping Time by Ship Mode ---
plt.subplot(2, 2, 1)  # First plot (top left)
shipping_mode_bar = shipping_mode_delay.plot(
    kind='bar', 
    title='Average Shipping Time by Ship Mode', 
    ylabel='Days', 
    color='skyblue', 
    ax=plt.gca()
)
plt.xticks(rotation=45)  # Rotate x-axis labels
for bar in shipping_mode_bar.patches:
    shipping_mode_bar.annotate(
        round(bar.get_height(), 1), 
        (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
        ha='center', va='bottom'
    )

# --- 2. Average Shipping Time by Region ---
plt.subplot(2, 2, 2)  # Second plot (top right)
region_bar = region_delay.plot(
    kind='bar', 
    title='Average Shipping Time by Region', 
    ylabel='Days', 
    color='lightgreen', 
    ax=plt.gca()
)
plt.xticks(rotation=45)  # Rotate x-axis labels
for bar in region_bar.patches:
    region_bar.annotate(
        round(bar.get_height(), 1), 
        (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
        ha='center', va='bottom'
    )

# --- 3. Average Shipping Time by Product Category ---
plt.subplot(2, 2, 3)  # Third plot (bottom left)
category_bar = category_delay.plot(
    kind='bar', 
    title='Average Shipping Time by Product Category', 
    ylabel='Days', 
    color='lightcoral', 
    ax=plt.gca()
)
plt.xticks(rotation=45)  # Rotate x-axis labels
for bar in category_bar.patches:
    category_bar.annotate(
        round(bar.get_height(), 1), 
        (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
        ha='center', va='bottom'
    )

# --- Display the entire dashboard ---
plt.tight_layout()
plt.show()

# #######################

max_shipping_time = {
    'Standard Class': 5,
    'First Class': 3,
    'Second Class': 8,
    'Same Day': 1
}

df['Is Delayed'] = df.apply(lambda row: row['Shipping Time'] > max_shipping_time[row['Ship Mode']], axis=1)

delayed_customers = df[df['Is Delayed']]['Customer ID'].unique()

delayed_orders_df = df[df['Customer ID'].isin(delayed_customers)]


repeated_customers_after_delay = delayed_orders_df.groupby('Customer ID').size().reset_index(name='Purchase Count After Delay')
repeated_customers_after_delay = repeated_customers_after_delay[repeated_customers_after_delay['Purchase Count After Delay'] > 1]

repeated_customers_info = pd.merge(repeated_customers_after_delay, df[['Customer ID', 'Customer Name']].drop_duplicates(), on='Customer ID')

purchase_count_before_delay = df[~df['Is Delayed']].groupby('Customer ID').size().reset_index(name='Purchase Count Before Delay')
top_customers_info = pd.merge(repeated_customers_info, purchase_count_before_delay, on='Customer ID')


top_customers = top_customers_info.nlargest(20, 'Purchase Count After Delay')

plt.figure(figsize=(10, 6))

top_customers.set_index('Customer Name')[['Purchase Count Before Delay', 'Purchase Count After Delay']].plot(
    kind='barh', 
    stacked=False, 
    color=['#FF4500', '#1E90FF']  
)


plt.title('Top 20 Customers: Purchases Before and After Delay')
plt.xlabel('Number of Purchases')
plt.ylabel('Customer Name')

plt.tight_layout()


plt.show()





avg_sales = df.groupby(["Customer ID", "Customer Name"])[['Sales']].agg(["mean"]).reset_index()

avg_sales.columns = ['Customer ID', 'Customer Name', 'Mean Sales']


top_customers = avg_sales.sort_values('Mean Sales', ascending=False).head(10)


purchase_counts = df.groupby(["Customer ID", "Customer Name"]).size().reset_index(name='Purchase Count')

top_customers_with_count = pd.merge(top_customers, purchase_counts, on=["Customer ID", "Customer Name"])

top_customers_by_purchase = purchase_counts.sort_values('Purchase Count', ascending=False).head(10)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6)) 

top_customers_with_count.plot(x='Customer Name', y='Mean Sales', kind="bar", color='lightblue', ax=ax1, position=0, width=0.4, legend=False)
ax1.set_ylabel('Average Sales', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title('Top 10 Customers by Average Sales')

ax3 = ax1.twinx()
top_customers_with_count.plot(x='Customer Name', y='Purchase Count', kind="bar", color='lightgreen', ax=ax3, position=1, width=0.4, legend=False)
ax3.set_ylabel('Purchase Count', color='green')
ax3.tick_params(axis='y', labelcolor='green')

top_customers_by_purchase.plot(x='Customer Name', y='Purchase Count', kind="bar", color='lightgreen', ax=ax2, legend=False)
ax2.set_title('Top 10 Customers by Purchase Count')
ax2.set_ylabel('Purchase Count')
ax2.set_xlabel('Customer Name')
ax2.tick_params(axis='x', rotation=45)


plt.tight_layout()


plt.show()


# Fifth Dashboard

# Group data by category and calculate total sales
category_sales = df.groupby('Category')[['Sales']].sum().sort_values('Sales', ascending=False)

# Group data by region and calculate total sales
region_sales = df.groupby('Region')[['Sales']].sum().sort_values('Sales', ascending=False)

# Group data by sub-category and calculate total sales
subcategory_sales = df.groupby('Sub-Category')[['Sales']].sum().sort_values('Sales', ascending=False)

# Group data by product and region, calculating total sales
product_region_sales = df.groupby(['Product Name', 'Region'])[['Sales']].sum().sort_values('Sales', ascending=False)

# Select a specific number of top-selling products (for example, top 10)
top_products = product_region_sales.groupby(level=0).sum()['Sales'].nlargest(10).index
filtered_products_by_region = df[df['Product Name'].isin(top_products)].groupby(['Region', 'Product Name']).size().unstack(fill_value=0)

# Set up the dashboard
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.4)  # Fix the syntax error here

# Plot total sales by category
category_bar = category_sales.plot(kind='bar', ax=axs[0, 0], title='Total Sales by Category', legend=False, color='skyblue')
axs[0, 0].set_ylabel('Total Sales')

for bar in category_bar.patches:
    category_bar.annotate(round(bar.get_height(), 2), 
                          (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                          ha='center', va='bottom')

# Plot total sales by region
region_bar = region_sales.plot(kind='bar', ax=axs[0, 1], title='Total Sales by Region', legend=False, color='lightgreen')
axs[0, 1].set_ylabel('Total Sales')

for bar in region_bar.patches:
    region_bar.annotate(round(bar.get_height(), 2), 
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                        ha='center', va='bottom')

# Plot total sales by sub-category without values on top of the bars
subcategory_bar = subcategory_sales.plot(kind='bar', ax=axs[1, 0], title='Total Sales by Sub-Category', legend=False, color='salmon')
axs[1, 0].set_ylabel('Total Sales')

# Plot a stacked bar chart showing the number of products sold in each region
filtered_products_by_region.plot(kind='bar', stacked=True, ax=axs[1, 1], figsize=(13, 8))

# Configure the stacked chart
axs[1, 1].set_title('Top 10 Products Sales by Region')
axs[1, 1].set_ylabel('Number of Products Sold')
axs[1, 1].set_xlabel('Region')
axs[1, 1].legend(title='Product Name', bbox_to_anchor=(1.05, 1), loc='upper left')
axs[1, 1].set_xticklabels(axs[1, 1].get_xticklabels(), rotation=45)

# Display the entire dashboard
plt.tight_layout()
plt.show()



# Sixth Dashboard

# Prepare data
customer_shipping = df.groupby(['Customer ID', 'Ship Mode']).size().unstack(fill_value=0)

# Analyze customer preferences based on shipping mode
shipping_preferences = df.groupby('Ship Mode')['Customer ID'].nunique()

# Analyze average sales by shipping mode
avg_sales_by_shipping = df.groupby('Ship Mode')['Sales'].mean()

# Define maximum values for each shipping method
max_shipping_time = {
    'Standard Class': 5,  # Max shipping time for Standard Class is 5 days
    'First Class': 3,     # Max shipping time for First Class is 3 days
    'Second Class': 8,    # Max shipping time for Second Class is 8 days
    'Same Day': 1         # Max shipping time for Same Day is 1 day
}

# Create a column to identify whether the order is delayed based on shipping method
df['Is Delayed'] = df.apply(lambda row: row['Shipping Time'] > max_shipping_time[row['Ship Mode']], axis=1)

# Calculate the number of delayed orders for each shipping mode
delayed_shipping_counts = df[df['Is Delayed']].groupby('Ship Mode').size()

# Set up the dashboard with three charts
fig, axs = plt.subplots(1, 3, figsize=(13, 6))  # 3 columns in one row

# --- 1. Pie chart showing customer preferences for shipping modes ---
axs[0].pie(shipping_preferences, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral', 'lightpink'])
axs[0].set_title('Customer Shipping Mode Preferences')

# --- 2. Bar chart showing average sales by shipping mode ---
avg_sales_bar = avg_sales_by_shipping.plot(kind='bar', ax=axs[1], color='skyblue')
axs[1].set_title('Average Sales by Shipping Mode')
axs[1].set_ylabel('Average Sales')
axs[1].set_xlabel('Shipping Mode')
axs[1].set_xticklabels(avg_sales_by_shipping.index, rotation=45)

# Add values above bars for the average sales chart
for bar in avg_sales_bar.patches:
    avg_sales_bar.annotate(f'{bar.get_height():.2f}', 
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                            ha='center', va='bottom')

# --- 3. Bar chart showing delayed orders by shipping mode ---
delayed_shipping_bar = delayed_shipping_counts.plot(kind='bar', ax=axs[2], color='lightblue')
axs[2].set_title('Delayed Orders by Ship Mode')
axs[2].set_ylabel('Number of Delayed Orders')
axs[2].set_xlabel('Shipping Mode')
axs[2].set_xticklabels(delayed_shipping_counts.index, rotation=45)

# Add values above bars for the delayed orders chart
for bar in delayed_shipping_bar.patches:
    delayed_shipping_bar.annotate(int(bar.get_height()), 
                                  (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                                  ha='center', va='bottom')

# Improve layout of the dashboard
plt.tight_layout()
plt.show()




# seven dash board

# Prepare data for shipping mode usage by region and category
region_shipping = df.groupby(['Region', 'Ship Mode']).size().unstack(fill_value=0)
category_shipping = df.groupby(['Category', 'Ship Mode']).size().unstack(fill_value=0)

# Set up the dashboard with two subplots (stacked bar charts)
fig2, axs2 = plt.subplots(2, 1, figsize=(13, 6))

# --- 1. Stacked Bar Chart: Shipping Mode Usage by Region ---
region_shipping_bar = region_shipping.plot(kind='bar', stacked=True, ax=axs2[0])
axs2[0].set_title('Shipping Mode Usage by Region')
axs2[0].set_ylabel('Number of Shipments')
axs2[0].set_xlabel('Region')
axs2[0].set_xticklabels(region_shipping.index, rotation=45)

# Add value annotations on the bars
for bar in region_shipping_bar.patches:
    region_shipping_bar.annotate(f'{bar.get_height()}', 
                                 (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                                 ha='center', va='bottom')

# --- 2. Stacked Bar Chart: Shipping Mode Usage by Category ---
category_shipping_bar = category_shipping.plot(kind='bar', stacked=True, ax=axs2[1])
axs2[1].set_title('Shipping Mode Usage by Category')
axs2[1].set_ylabel('Number of Shipments')
axs2[1].set_xlabel('Category')
axs2[1].set_xticklabels(category_shipping.index, rotation=45)

# Add value annotations on the bars
for bar in category_shipping_bar.patches:
    category_shipping_bar.annotate(f'{bar.get_height()}', 
                                   (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                                   ha='center', va='bottom')

# Adjust layout for better visualization
plt.tight_layout()
plt.show()




# eights dashboard

# Calculate total sales by category and sub-category
sales_by_category = df.groupby('Category')['Sales'].sum().sort_values(ascending=True)
sales_by_subcategory = df.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=True)

# Set up the dashboard with two subplots for category and sub-category sales
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# --- 1. Bar Chart: Sales by Category ---
axs[0].bar(sales_by_category.index, sales_by_category, color='lightblue')
axs[0].set_title('Sales by Category')
axs[0].set_ylabel('Total Sales')
axs[0].set_xlabel('Category')
axs[0].tick_params(axis='x', rotation=45)

# --- 2. Bar Chart: Sales by Sub-Category ---
axs[1].bar(sales_by_subcategory.index, sales_by_subcategory, color='lightcoral')
axs[1].set_title('Sales by Sub-Category')
axs[1].set_ylabel('Total Sales')
axs[1].set_xlabel('Sub-Category')
axs[1].tick_params(axis='x', rotation=45)

# Adjust layout for better visualization
plt.tight_layout()
plt.show()








# Insights and Recommendations:

# 1. Sector Dominance: 
# The "Consumer" sector drives the majority of sales. This suggests a potential growth opportunity 
# in targeting promotional offers or customized deals towards the "Corporate" and "Home Office" sectors.

# 2. Sales Growth Opportunity:
# Sales may show significant growth during the holiday season. Consider ramping up marketing efforts 
# or offering discounts during peak seasons to maximize revenue.

# 3. Top Product Categories:
# Technology products are the best-sellers. Consider increasing stock and promoting these items more aggressively.

# 4. Regional Strategy:
# The "West" region outperforms other areas. Focus on improving sales strategies in underperforming regions, 
# such as the "South," by crafting targeted promotions or sales campaigns.

# 5. Targeting Consumer Segments:
# The "Consumer" segment accounts for the highest sales. To expand the customer base, think about 
# creating marketing strategies specifically tailored to other segments like "Corporate" and "Home Office."

# These key insights can help guide strategic decisions around inventory management, regional sales focus, 
# and tailored marketing campaigns to drive growth in underperforming areas and sectors.


