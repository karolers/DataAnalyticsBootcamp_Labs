mysql
USE labs;

CREATE TABLE IF NOT EXISTS 
labs.salesperson (
staff_id INT PRIMARY KEY,
name VARCHAR(40),
store VARCHAR(40)
);

CREATE TABLE IF NOT EXISTS 
labs.cars (
car_id INT PRIMARY KEY,
vin VARCHAR(40),
manufacturer VARCHAR(40),
model VARCHAR(40),
year int,
color VARCHAR(40)
);

CREATE TABLE IF NOT EXISTS 
labs.customer (
id INT PRIMARY KEY,
name VARCHAR(40),
phone VARCHAR(40),
email VARCHAR(40),
adress VARCHAR(40),
city VARCHAR(40),
state VARCHAR(40),
country VARCHAR(40),
zip_code VARCHAR(40)
);

CREATE TABLE IF NOT EXISTS 
labs.invoices (
invoice_id INT PRIMARY KEY,
date VARCHAR(40),
car_id INT,
customer_id INT,
staff_id INT
);