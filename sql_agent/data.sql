INSERT INTO Restaurants (Name, Location, CuisineType, Rating, HoursOfOperation)
VALUES
('Lahore Tikka House', 'Lahore', 'Pakistani', 4.5, '9:00 AM - 11:00 PM'),
('Karachi Broast', 'Karachi', 'Fast Food', 4.2, '10:00 AM - 12:00 AM'),
('Peshawar Chapli Kebab', 'Peshawar', 'Afghani', 4.7, '8:00 AM - 10:00 PM'),
('Islamabad Café', 'Islamabad', 'Continental', 4.0, '7:00 AM - 9:00 PM'),
('Hyderabad Biryani Corner', 'Hyderabad', 'Biryani', 4.6, '11:00 AM - 11:00 PM'),
('Quetta Sajji House', 'Quetta', 'Barbecue', 4.3, '12:00 PM - 10:00 PM');

INSERT INTO Customers (Name, Email, Phone, Address, PaymentInfo, LoyaltyPoints)
VALUES
('Ali Ahmed', 'ali.ahmed@example.com', '03001234567', 'Gulberg, Lahore', 'Visa ****1234', 120),
('Ayesha Khan', 'ayesha.khan@example.com', '03021234567', 'DHA, Karachi', 'Easypaisa ****5678', 90),
('Zainab Ali', 'zainab.ali@example.com', '03131234567', 'F-7, Islamabad', 'COD', 200),
('Bilal Shaikh', 'bilal.shaikh@example.com', '03041234567', 'Cantt, Peshawar', 'Visa ****4321', 50),
('Hassan Raza', 'hassan.raza@example.com', '03251234567', 'University Road, Karachi', 'JazzCash ****9876', 75),
('Sara Qureshi', 'sara.qureshi@example.com', '03361234567', 'G-11, Islamabad', 'COD', 180),
('Umar Siddiqui', 'umar.siddiqui@example.com', '03471234567', 'Satellite Town, Quetta', 'Easypaisa ****3456', 140),
('Fatima Tariq', 'fatima.tariq@example.com', '03081234567', 'Gulshan-e-Iqbal, Karachi', 'Visa ****6543', 160),
('Fahad Ali', 'fahad.ali@example.com', '03191234567', 'Blue Area, Islamabad', 'COD', 85),
('Sana Malik', 'sana.malik@example.com', '03451234567', 'Model Town, Lahore', 'JazzCash ****1122', 50),
('Rizwan Butt', 'rizwan.butt@example.com', '03051234567', 'Faisal Town, Lahore', 'COD', 110),
('Farah Javed', 'farah.javed@example.com', '03211234567', 'Garden Town, Lahore', 'Easypaisa ****2211', 190);

INSERT INTO Orders (CustomerID, RestaurantID, OrderStatus, TotalCost, DeliveryAddress, PaymentMethod, SpecialInstructions)
VALUES
(1, 1, 'Completed', 1200.50, 'Gulberg, Lahore', 'Visa', 'No onions please'),
(2, 2, 'In Progress', 1500.75, 'DHA, Karachi', 'Easypaisa', 'Extra spicy'),
(3, 3, 'Rejected', 800.25, 'F-7, Islamabad', 'COD', 'No special instructions'),
(4, 4, 'Completed', 2000.00, 'Cantt, Peshawar', 'Visa', 'Add extra sauce'),
(5, 5, 'Completed', 950.00, 'University Road, Karachi', 'JazzCash', 'Less oil'),
(6, 6, 'In Progress', 1350.60, 'G-11, Islamabad', 'COD', 'No special instructions'),
(7, 1, 'In Progress', 1750.30, 'Satellite Town, Quetta', 'Easypaisa', 'Gluten-free'),
(8, 2, 'Completed', 2100.80, 'Gulshan-e-Iqbal, Karachi', 'Visa', 'No cheese'),
(9, 3, 'Rejected', 1150.40, 'Blue Area, Islamabad', 'COD', 'Extra ketchup'),
(10, 4, 'Completed', 1250.90, 'Model Town, Lahore', 'JazzCash', 'Extra napkins'),
(11, 5, 'In Progress', 950.50, 'Faisal Town, Lahore', 'COD', 'Less salt'),
(12, 6, 'Completed', 1450.20, 'Garden Town, Lahore', 'Easypaisa', 'No instructions'),
(1, 3, 'Completed', 1650.75, 'Gulberg, Lahore', 'Visa', 'Less spicy'),
(2, 5, 'Completed', 2250.90, 'DHA, Karachi', 'JazzCash', 'No sugar'),
(3, 1, 'In Progress', 1250.50, 'F-7, Islamabad', 'COD', 'Add garlic bread'),
(4, 2, 'Completed', 1400.00, 'Cantt, Peshawar', 'Easypaisa', 'More cheese'),
(5, 3, 'Completed', 1850.75, 'University Road, Karachi', 'Visa', 'Extra onions'),
(6, 4, 'In Progress', 1050.30, 'G-11, Islamabad', 'COD', 'No tomatoes'),
(7, 6, 'Rejected', 2150.60, 'Satellite Town, Quetta', 'Easypaisa', 'Spicy, please'),
(8, 1, 'Completed', 900.90, 'Gulshan-e-Iqbal, Karachi', 'JazzCash', 'Extra napkins');

INSERT INTO Payments (OrderID, PaymentStatus, PaymentMethod)
VALUES
(1, 'Paid', 'Visa'),
(2, 'Pending', 'Easypaisa'),
(3, 'Failed', 'COD'),
(4, 'Paid', 'Visa'),
(5, 'Paid', 'JazzCash'),
(6, 'Pending', 'COD'),
(7, 'Pending', 'Easypaisa'),
(8, 'Paid', 'Visa'),
(9, 'Failed', 'COD'),
(10, 'Paid', 'JazzCash'),
(11, 'Pending', 'COD'),
(12, 'Paid', 'Easypaisa'),
(13, 'Paid', 'Visa'),
(14, 'Paid', 'JazzCash'),
(15, 'Pending', 'COD'),
(16, 'Paid', 'Easypaisa'),
(17, 'Paid', 'Visa'),
(18, 'Pending', 'COD'),
(19, 'Failed', 'Easypaisa'),
(20, 'Paid', 'JazzCash');

-- Re-populate the OrderItems table with references to FoodItems
INSERT INTO OrderItems (OrderID, FoodItemID, Quantity, Price)
VALUES
-- Orders from Lahore Tikka House (RestaurantID = 1)
(1, 1, 2, 1000),  -- 2 Chicken Tikkas
(1, 3, 3, 150),   -- 3 Tandoori Roti
(2, 2, 1, 400),   -- 1 Beef Seekh Kebab

-- Orders from Karachi Broast (RestaurantID = 2)
(3, 6, 1, 600),   -- 1 Broast Chicken
(4, 8, 2, 800),   -- 2 Zinger Burgers
(4, 7, 1, 200),   -- 1 Fries

-- Orders from Peshawar Chapli Kebab (RestaurantID = 3)
(5, 11, 2, 1000), -- 2 Chapli Kebabs
(6, 12, 1, 100),  -- 1 Afghani Naan

-- Orders from Islamabad Café (RestaurantID = 4)
(7, 14, 1, 600),  -- 1 Fish and Chips
(8, 13, 1, 400),  -- 1 Grilled Chicken Sandwich
(8, 16, 1, 300),  -- 1 Cheese Cake

-- Orders from Hyderabad Biryani Corner (RestaurantID = 5)
(9, 18, 1, 400),  -- 1 Chicken Biryani
(10, 19, 1, 450), -- 1 Beef Biryani

-- Orders from Quetta Sajji House (RestaurantID = 6)
(11, 21, 1, 800), -- 1 Chicken Sajji
(12, 22, 1, 1200),-- 1 Mutton Sajji
(12, 23, 2, 120), -- 2 Balochi Roti
(13, 24, 1, 150); -- 1 Jalebi

INSERT INTO DeliveryPersonnel (AssignedOrderIDs, Name, ContactInformation, CurrentLocation)
VALUES
(ARRAY[1, 4, 5], 'Ahmed Ali', '0300-1234567', 'Gulberg, Lahore'),
(ARRAY[2, 8, 6], 'Imran Khan', '0320-1234567', 'DHA, Karachi'),
(ARRAY[3, 9, 7], 'Asad Khan', '0301-1234567', 'Blue Area, Islamabad'),
(ARRAY[10, 12], 'Bilal Ahmed', '0340-1234567', 'Model Town, Lahore'),
(ARRAY[11, 13], 'Hassan Tariq', '0333-1234567', 'Faisal Town, Lahore'),
(ARRAY[14, 15], 'Farooq Malik', '0302-1234567', 'G-11, Islamabad'),
(ARRAY[16, 17], 'Umar Zafar', '0321-1234567', 'Satellite Town, Quetta'),
(ARRAY[18, 19], 'Naveed Qureshi', '0345-1234567', 'University Road, Karachi'),
(ARRAY[20], 'Salman Butt', '0312-1234567', 'Cantt, Peshawar'),
(ARRAY[1, 2, 3], 'Zeeshan Ali', '0309-1234567', 'Gulshan-e-Iqbal, Karachi');

INSERT INTO FoodItems (RestaurantID, Name, Description, Price, Category)
VALUES
-- Lahore Tikka House Menu
(1, 'Chicken Tikka', 'Grilled chicken marinated with spices', 500, 'Main Course'),
(1, 'Beef Seekh Kebab', 'Minced beef skewers with traditional spices', 400, 'Main Course'),
(1, 'Tandoori Roti', 'Traditional clay oven bread', 50, 'Appetizer'),
(1, 'Raita', 'Yogurt with spices and herbs', 80, 'Appetizer'),
(1, 'Gulab Jamun', 'Deep fried milk-based dessert', 150, 'Dessert'),

-- Karachi Broast Menu
(2, 'Broast Chicken', 'Crispy fried chicken with a special sauce', 600, 'Main Course'),
(2, 'Fries', 'Crispy golden fries', 200, 'Appetizer'),
(2, 'Spicy Wings', 'Deep fried spicy chicken wings', 350, 'Appetizer'),
(2, 'Zinger Burger', 'Fried chicken burger with a spicy twist', 400, 'Main Course'),
(2, 'Chocolate Lava Cake', 'Warm chocolate cake with gooey center', 250, 'Dessert'),

-- Peshawar Chapli Kebab Menu
(3, 'Chapli Kebab', 'Fried minced meat patty with spices', 500, 'Main Course'),
(3, 'Afghani Naan', 'Traditional Afghan flatbread', 100, 'Appetizer'),
(3, 'Kabuli Pulao', 'Rice with lamb, raisins, and carrots', 750, 'Main Course'),
(3, 'Mint Chutney', 'Spicy green chutney made from fresh mint', 100, 'Appetizer'),
(3, 'Baklava', 'Layered pastry with nuts and honey', 300, 'Dessert'),

-- Islamabad Café Menu
(4, 'Grilled Chicken Sandwich', 'Grilled chicken with lettuce, tomato, and mayo', 400, 'Main Course'),
(4, 'Caesar Salad', 'Fresh lettuce, parmesan, and croutons', 350, 'Appetizer'),
(4, 'Fish and Chips', 'Crispy fried fish with potato fries', 600, 'Main Course'),
(4, 'Cheese Cake', 'Rich creamy cheesecake with a graham cracker crust', 300, 'Dessert'),
(4, 'Espresso', 'Shot of rich and bold espresso', 150, 'Beverage'),

-- Hyderabad Biryani Corner Menu
(5, 'Chicken Biryani', 'Spiced rice with marinated chicken', 400, 'Main Course'),
(5, 'Beef Biryani', 'Spiced rice with marinated beef', 450, 'Main Course'),
(5, 'Shami Kebab', 'Minced beef patty with spices', 120, 'Appetizer'),
(5, 'Raita', 'Yogurt with spices and herbs', 80, 'Appetizer'),
(5, 'Kheer', 'Traditional rice pudding with nuts', 150, 'Dessert'),

-- Quetta Sajji House Menu
(6, 'Chicken Sajji', 'Whole chicken marinated in spices and roasted', 800, 'Main Course'),
(6, 'Mutton Sajji', 'Whole mutton leg marinated in spices and roasted', 1200, 'Main Course'),
(6, 'Balochi Roti', 'Traditional bread from Balochistan', 60, 'Appetizer'),
(6, 'Chutney', 'Spicy chutney made from chili and tamarind', 50, 'Appetizer'),
(6, 'Jalebi', 'Deep fried sugary dessert', 150, 'Dessert');