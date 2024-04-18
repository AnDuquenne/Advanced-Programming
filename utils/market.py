class Position:
    def __init__(self, opening_price: float, opening_time, amount: float, direction: str, closing_price: float):

        assert direction in ['long', 'short'], "Direction must be either long or short"

        self._status = 'open'
        self._opening_price = opening_price
        self._opening_time = opening_time
        self._amount = amount
        self._direction = direction
        self._closing_price = closing_price
        self._closing_time = None
        self._profit = 0

    def close(self, closing_price, closing_time):
        """
        Close the position
        :param closing_price: the closing price
        :param closing_time: the closing time
        :return:
        """
        self._status = 'closed'
        self._closing_price = closing_price
        self._closing_time = closing_time
        if self._direction == 'long':
            self._profit = self._amount * (self._closing_price - self._opening_price)
        else:
            self._profit = - (self._amount * (self._closing_price - self._opening_price))

    def dollars_value(self, current_price):
        """
        Calculate the value of the position in dollars
        :param current_price: The actual price of the asset
        :return:
        """
        if self._direction == 'long':
            return self._amount * current_price
        else:
            return - self._amount * current_price

    def pnl(self, current_price):
        """
        Calculate the profit and loss of the position (to be used when the position is still open)
        :param current_price: The actual price of the asset
        :return:
        """
        if self._direction == 'long':
            return self._amount * (current_price - self._opening_price)
        else:
            return - self._amount * (current_price - self._opening_price)

    @property
    def opening_price(self):
        return self._opening_price

    @opening_price.setter
    def opening_price(self, value):
        if isinstance(value, float):
            self._opening_price = value
        else:
            raise ValueError("Opening price must be a float")

    @property
    def closing_price(self):
        return self._closing_price

    @closing_price.setter
    def closing_price(self, value):
        if isinstance(value, float):
            self._closing_price = value
        else:
            raise ValueError("Closing price must be a float")

    @property
    def profit(self):
        return self._profit

    @profit.setter
    def profit(self, value):
        if isinstance(value, float):
            self._profit = value
        else:
            raise ValueError("Profit must be a float")

    @property
    def amount(self):
        return self._amount

    @amount.setter
    def amount(self, value):
        if isinstance(value, int):
            self._amount = value
        else:
            raise ValueError("Amount must be an integer")

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
        if value in ['long', 'short']:
            self._direction = value
        else:
            raise ValueError("Direction must be either long or short")

    @property
    def opening_time(self):
        return self._opening_time

    @opening_time.setter
    def opening_time(self, value):
        if isinstance(value, float):
            self._opening_time = value
        else:
            raise ValueError("Opening time must be a float")

    @property
    def closing_time(self):
        return self._closing_time

    @closing_time.setter
    def closing_time(self, value):
        if isinstance(value, float):
            self._closing_time = value
        else:
            raise ValueError("Closing time must be a float")

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        if value in ['open', 'closed']:
            self._status = value
        else:
            raise ValueError("Status must be either open or closed")

    def __str__(self):
        return f'Opening price: {self._opening_price}, Closing price: {self._closing_price}, Profit: {self._profit}'


class Order:
    def __init__(self, time, price: float, amount: float, direction: str):
        self._time = time
        self._price = price
        self._amount = amount
        self._direction = direction

    def __str__(self):
        return f'Time: {self._time}, Amount: {self._amount}, Direction: {self._direction}'

    @property
    def amount(self):
        return self._amount

    @amount.setter
    def amount(self, value):
        if isinstance(value, int):
            self._amount = value
        else:
            raise ValueError("Amount must be an integer")

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
        if value in ['long', 'short']:
            self._direction = value
        else:
            raise ValueError("Direction must be either long or short")

    @property
    def price(self):
        return self._price

    @price.setter
    def price(self, value):
        if isinstance(value, float):
            self._price = value
        else:
            raise ValueError("Price must be a float")

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        if isinstance(value, float):
            self._time = value
        else:
            raise ValueError("Time must be a float")

