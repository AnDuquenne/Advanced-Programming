
class Position:
    def __init__(self, opening_price, opening_time, amount, direction, closing_price):
        self.status = 'open'
        self.opening_price = opening_price
        self.opening_time = opening_time
        self.amount = amount
        self.direction = direction
        self.closing_price = closing_price
        self.closing_time = None

    def close(self, closing_price, closing_time):
        self.status = 'closed'
        self.closing_price = closing_price
        self.closing_time = closing_time
        profit_ratio = (self.closing_price - self.opening_price) / self.opening_price
        profit = self.amount * (1 + profit_ratio)

        return profit

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

