from django.contrib import admin
from . models import Exam
from django.urls import reverse
from django.utils.safestring import mark_safe   
from django.contrib.contenttypes.admin import GenericTabularInline

"""
class QuestionInline(admin.TabularInline):
    model = Question
    extra = 0
    max_num = 0
    fields = ['creator_user_id','question_text', 'question_type', 'question_level', 'question_category', 'status']
    can_delete = False
    
# Register your models here.
class ExamAdmin(admin.ModelAdmin):
    actions = None
    list_display=('exam_name','status','created_at',)

    inlines = [QuestionInline,]

    def has_delete_permission(self, request, obj=None):
        # Disable delete
        return True

    def has_add_permission(self, request):
        return True

    def has_delete_permission(self, request, obj=None):
        return True

    def has_change_permission(self, request, obj=None):
        return True

#class QuestionAdmin(admin.ModelAdmin):                                           
#    pass    
admin.site.register(Exam,ExamAdmin)
#admin.site.register(Question,QuestionAdmin)
"""