
class Position:
    def __init__(self, opening_price, opening_time, amount, direction, closing_price):
        self.status = 'open'
        self.opening_price = opening_price
        self.opening_time = opening_time
        self.amount = amount
        self.direction = direction
        self.closing_price = closing_price
        self.closing_time = None
        self.profit = 0

    def close(self, closing_price, closing_time):
        self.status = 'closed'
        self.closing_price = closing_price
        self.closing_time = closing_time
        profit_ratio = (self.closing_price - self.opening_price) / self.opening_price
        self.profit = self.amount * (1 + profit_ratio)

    @property
    def opening_price(self):
        return self.opening_price

    @property
    def closing_price(self):
        return self.closing_price

    @property
    def profit(self):
        return self.profit

    @property
    def amount(self):
        return self.amount

    @property
    def direction(self):
        return self.direction

    @property
    def opening_time(self):
        return self.opening_time

    @property
    def closing_time(self):
        return self.closing_time

    @property
    def status(self):
        return self.status


    def __str__(self):
        return f'Opening price: {self.opening_price}, Closing price: {self.closing_price}, Profit: {self.profit}'


class Order:
    def __init__(self, time, price, amount, direction):
        self.time = time
        self.price = price
        self.amount = amount
        self.direction = direction

    def __str__(self):
        return f'Time: {self.time}, Amount: {self.amount}, Direction: {self.direction}'

    @property
    def amount(self):
        return self.amount

    @property
    def direction(self):
        return self.direction

    @property
    def price(self):
        return self.price

    @property
    def time(self):
        return self.time