from app import db

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), index=True, unique=False)
    file_path = db.Column(db.String(300), index=True, unique=True)

    def __repr__(self):
        return '<User {}>'.format(self.name) 