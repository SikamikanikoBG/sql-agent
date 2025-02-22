CREATE TABLE employees (
    id INT PRIMARY KEY AUTOINCREMENT,
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    salary DECIMAL(10, 2),
    city VARCHAR(255)
);

CREATE TABLE projects (
    id INT PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT CHECK (length <= 300),
    status ENUM('active', 'inactive') DEFAULT 'active'
);

INSERT INTO employees VALUES ('Smith', 'John', 'New York', 50000);
INSERT INTO employees VALUES ('Johnson', 'Doe', 'Los Angeles', 60000);
INSERT INTO employees VALUES ('Williams', 'Kettle', 'Chicago', 45000);

INSERT INTO projects VALUES ('Project 1', 'Software Development', 'Detailed analysis of user requirements and system design', 'active');
INSERT INTO projects VALUES ('Project 2', 'Web Development', 'Building a modern web application', 'inactive');
INSERT INTO projects VALUES ('Project 3', 'Data Analysis', 'Extracting insights from large datasets', 'active');

CREATE TABLE sales (
    id INT PRIMARY KEY AUTOINCREMENT,
    customer_id INT,
    amount DECIMAL(10, 2),
    created_date DATE,
    FOREIGN KEY (customer_id) REFERENCES employees(id)
);

INSERT INTO sales VALUES (1, NULL, 750.00, CURRENT_DATE);
INSERT INTO sales VALUES (2, 1, 600.00, CURRENT_DATE - 1);
INSERT INTO sales VALUES (3, 2, 450.00, CURRENT_DATE - 2);
