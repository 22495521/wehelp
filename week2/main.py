import math


class Point:
    def __init__(self,x , y):
        self.x = x
        self.y = y

class Line:
    def __init__(self,p1:Point ,p2:Point):
        self.p1 = p1
        self.p2 = p2

    def direction(self):
        return (self.p2.x - self.p1.x, self.p2.y - self.p1.y) 
    
    def is_parallel_to(self,other:Line):
        dx1, dy1 = self.direction()
        dx2, dy2 = other.direction()
        det = dx1 * dy2 - dx2 * dy1

        return det == 0
    
    def is_perpendicular_to(self, other: Line):
        dx1, dy1 = self.direction()
        dx2, dy2 = other.direction()
        dot = dx1 * dx2 + dy1 * dy2
        return dot == 0


class Circle:
    pi = math.pi

    def __init__(self,r,center:Point):
        self.radius = r
        self.center = center

    def area(self):
        return (self.radius ** 2) * Circle.pi
    
    def distance_to(self, other: Circle):
        dx = other.center.x - self.center.x
        dy = other.center.y - self.center.y
        return math.sqrt(dx ** 2 + dy ** 2)
    
    def intersects(self, other: Circle):
        d = self.distance_to(other)
        r_sum = self.radius + other.radius
        r_diff = abs(self.radius - other.radius)
        return r_diff <= d <= r_sum


class Polygon:
    def __init__(self, pl: list[Point]):
        self.pointList = pl

    def calculatePerimeter(self):
        
        perimeter = 0
        for index,i in enumerate(self.pointList):
            if index < len(self.pointList)-1:
                perimeter += Polygon.distance_to(i,self.pointList[index+1])
            else:
                perimeter += Polygon.distance_to(i,self.pointList[0])
        return  perimeter 

    @staticmethod
    def distance_to(pointA:Point, pointB:Point):
        dx = pointA.x - pointB.x
        dy = pointA.y - pointB.y
        return math.sqrt(dx ** 2 + dy ** 2)


p1 = Point(2,4)
p2 = Point(-6,1)
p3 = Point(2,2)
p4 = Point(-6,-1)
p5 = Point(-1,6)
p6 = Point(-4,-4)

LineA = Line(p1,p2) 
LineB = Line(p3,p4)
LineC = Line(p5,p6)

# 問題一
print("Are Line A and Line B parallel?",LineA.is_parallel_to(LineB))
# 問題二
print("Are Line C and Line A perpendicular?",LineA.is_perpendicular_to(LineC))

# 問題三
p7 = Point(6,3)
CircleA = Circle(2,p7)
print("Print the area of Circle A.",CircleA.area())

# 問題四
p8 = Point(8,1)
CircleB = Circle(1,p8)
print("Do Circle A and Circle B intersect?", CircleA.intersects(CircleB))



# 問題五
p9 = Point(2,0)
p10 = Point(5,-1)
p11 = Point(4,-4)
p12 = Point(-1,-2)

PolygonA = Polygon((p9,p10,p11,p12))
print('Print the perimeter of Polygon A.',PolygonA.calculatePerimeter())



print("======================")


class Red:
    def __init__(self,label,x,y,vx,vy):
        self.label = label
        self.point = 10
        self.x = x  
        self.y = y
        self.vx = vx
        self.vy = vy

    def walk(self):
        if self.point > 0:
            self.x += self.vx
            self.y += self.vy

    def beAttack(self,point):
        if self.point > 0:
            self.point -= point
            if(self.point <= 0):
                print(f"Dead {self.label} {self.x} {self.y}")


class BG:
    def __init__(self,x,y,point,range):
        self.x = x
        self.y = y
        self.point = point
        self.range = range


    def attack_to_area(self,enemy:Red):


        # 計算兩點距離
        distance = math.sqrt((self.x - enemy.x) ** 2 + (self.y - enemy.y) ** 2)
        if self.range >= distance :
            enemy.beAttack(self.point)

class Blue(BG):
    def __init__(self,x,y):
        super().__init__(x,y,1,2)

class Green(BG):
    def __init__(self,x,y):
        super().__init__(x,y,2,4)



E1 = Red("E1",-10,2,2,-1)
E2 = Red('E2',-8,0,3,1)
E3 = Red('E3',-9,-1,3,0)
T1 = Blue(-3,2)
T2 = Blue(-1,-2)
T3 = Blue(4,2)
T4 = Blue(7,0)
A1 = Green(1,1)
A2 = Green(4,-3)

eList = [E1,E2,E3]
TAList = [T1,T2,T3,T4,A1,A2]

for e in eList:
    for i in range(10):
        e.walk()
        for TA in TAList:
            TA.attack_to_area(e)

print(E1.label,E1.x,E1.y,E1.point)
print(E2.label ,E2.x,E2.y,E2.point)
print(E3.label,E3.x,E3.y,E3.point)
    