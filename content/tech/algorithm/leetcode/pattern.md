---
title: "Design"
date: 2024-05-10T09:47:15Z
lastmod: 2024-05-25
draft: true
description: ""
tags: ["design"]
# series: [""]
# series_order: 2
# layout: "simple"
showDate: true
---

## Singleton

饿汉

```c++
class Single
{
    public:
        static Single* GetInstance();
        static void deleteInstance();
        void print();
    private:
        //禁止外部构造和析构
        Single();
        ~Single();
        //禁止外部拷贝构造
        Single(const Single &signal)
        //禁止外部赋值操作
        const Single &operator=(const Single &signal);
    private:
        //唯一单实例对象指针
        static Single *p_Single;
};

Single* Single::p_Single = new Single();
Single* Single::GetInstance(){
    return p_Single;
}
void Single::deleteInstance(){
    if(p_Single){
        delete p_Single;
        p_Single = null;
    }
}
void Single::print(){
    std::cout<<"Instance ptr:"<<this<<endl;
}

```

Reference:
- [C++ 单例模式总结](https://blog.csdn.net/unonoi/article/details/121138176)