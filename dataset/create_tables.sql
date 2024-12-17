CREATE TABLE Restaurants (
    RestaurantID SERIAL PRIMARY KEY,
    Name VARCHAR(255) NOT NULL,
    Location VARCHAR(255),
    CuisineType VARCHAR(100),
    Rating FLOAT,
    HoursOfOperation VARCHAR(100)
);
CREATE TABLE Customers (
    CustomerID SERIAL PRIMARY KEY,
    Name VARCHAR(255) NOT NULL,
    Email VARCHAR(255) NOT NULL,
    Phone VARCHAR(20),
    Address VARCHAR(255),
    PaymentInfo VARCHAR(255),
    LoyaltyPoints INT,
    RegistrationDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE Orders (
    OrderID SERIAL PRIMARY KEY,
    CustomerID INT REFERENCES Customers(CustomerID),
    RestaurantID INT REFERENCES Restaurants(RestaurantID),
    OrderStatus VARCHAR(50),
    TotalCost FLOAT,
    DeliveryAddress VARCHAR(255),
    OrderDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PaymentMethod VARCHAR(100),
    SpecialInstructions TEXT
);
CREATE TABLE Payments (
    PaymentID SERIAL PRIMARY KEY,
    OrderID INT REFERENCES Orders(OrderID),
    PaymentStatus VARCHAR(100),
    TransactionDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PaymentMethod VARCHAR(100)
);
CREATE TABLE OrderItems (
    OrderItemID SERIAL PRIMARY KEY,
    OrderID INT REFERENCES Orders(OrderID),
    FoodItemID INT REFERENCES FoodItems(FoodItemID),
    Quantity INT NOT NULL,
    Price FLOAT NOT NULL -- Store the price at the time of order, in case of price changes
);
CREATE TABLE FoodItems (
    FoodItemID SERIAL PRIMARY KEY,
    RestaurantID INT REFERENCES Restaurants(RestaurantID),
    Name VARCHAR(255) NOT NULL,
    Description TEXT,
    Price FLOAT NOT NULL,
    Category VARCHAR(100)  -- e.g., Appetizer, Main Course, Dessert, etc.
);
CREATE TABLE DeliveryPersonnel (
    DeliveryID SERIAL PRIMARY KEY,
    AssignedOrderIDs INT[], -- Assuming Postgres array for multiple orders
    Name VARCHAR(255),
    ContactInformation VARCHAR(255),
    CurrentLocation VARCHAR(255)
);
CREATE TABLE FeedbackComplaints (
    FeedbackID SERIAL PRIMARY KEY,
    OrderID INT REFERENCES Orders(OrderID),
    CustomerID INT REFERENCES Customers(CustomerID),
    FeedbackText TEXT,
    Rating INT,
    DateSubmitted TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);