SET SQL_SAFE_UPDATES = 0;

UPDATE labs.salesperson
SET store = 'Miami'
WHERE name = 'Paige Turner';

UPDATE labs.customer
SET email = 'ppicasso@gmail.com'
WHERE name = 'Pablo Picasso';

UPDATE labs.customer
SET email = 'lincoln@us.gov'
WHERE name = 'Abraham Lincoln';

UPDATE labs.customer
SET email = 'hello@napoleon.me'
WHERE name = 'Napoléon Bonaparte';