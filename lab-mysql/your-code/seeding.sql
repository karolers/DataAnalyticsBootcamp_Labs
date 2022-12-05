SET sql_mode = "NO_AUTO_VALUE_ON_ZERO";

### Cars
INSERT INTO labs.cars (car_id, vin, manufacturer, model, year, color)
VALUES ROW(0, '3K096I98581DHSNUP', 'Volkswagen', 'Tiguan', 2019, 'Blue'),
    ROW(1, 'ZM8G7BEUQZ97IH46V', 'Peugeot', 'Rifter', 2019, 'Red'),
    ROW(2, 'RKXVNNIHLVVZOUB4M', 'Ford', 'Fusion', 2018, 'White'),
    ROW(3, 'HKNDGS7CU31E9Z7JW', 'Toyota', 'RAV4', 2018, 'Silver'),
    ROW(4, 'DAM41UDN3CHU2WVF6', 'Volvo', 'V60', 2019, 'Gray'),
    ROW(5, 'DAM41UDN3CHU2WVF6', 'Volvo', 'V60 Cross Country', 2019, 'Gray');
    
### Customers
INSERT INTO labs.customer (id, name, phone, email, adress, city, state, country, zip_code)
VALUES ROW(10001, 'Pablo Picasso', '+34 636 17 63 82', '-', 'Paseo de la Chopera, 14', 'Madrid', 'Madrid', 'Spain', 28045),
    ROW(20001, 'Abraham Lincoln', '+1 305 907 7086', '-', '120 SW 8th St', 'Miami', 'Florida', 'United States', 33130),
    ROW(30001, 'Napoléon Bonaparte', '+33 1 79 75 40 00', '-', '40 Rue du Colisée', 'Paris', 'Île-de-France', 'France', 75008);

### Salespersons
INSERT INTO labs.salesperson (staff_id, name, store)
VALUES ROW(00001, 'Petey Cruiser', 'Madrid'),
    ROW(00002, 'Anna Sthesia', 'Barcelona'),
    ROW(00003, 'Paul Molive', 'Berlin'),
    ROW(00004, 'Gail Forcewind', 'Paris'),
    ROW(00005, 'Paige Turner', 'Mimia'),
    ROW(00006, 'Bob Frapples', 'Mexico City'),
    ROW(00007, 'Walter Melon', 'Amsterdam'),
    ROW(00008, 'Shonda Leer', 'São Paulo');

### Invoices
INSERT INTO labs.invoices (invoice_id, date, car_id, customer_id, staff_id)
VALUES ROW(852399038, '22-08-2018', 0, 20001, 00004),
    ROW(731166526, '31-12-2018', 3, 10001, 00006),
    ROW(271135104, '22-01-2019', 2, 30001, 00008);